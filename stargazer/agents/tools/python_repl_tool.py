import functools
import logging
import multiprocessing
import re
import sys
import threading
import traceback
import ast
from io import StringIO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-local stdout capture
#
# sys.stdout is a process-wide global.  When the batch runner uses
# ThreadPoolExecutor, naively replacing sys.stdout in one thread pollutes
# all other threads' print() output.
#
# Fix: install a thin proxy once at module import time.  The proxy routes
# each write() to a per-thread capture buffer when one is active, or to the
# real stdout otherwise.  Other threads are completely unaffected.
# ---------------------------------------------------------------------------

_thread_local = threading.local()


class _ThreadLocalStdoutProxy:
    """Proxy for sys.stdout that supports per-thread output capture."""

    def __init__(self, real_stdout):
        # Store without triggering __setattr__ on this proxy
        object.__setattr__(self, "_real", real_stdout)

    def write(self, data: str) -> int:
        buf = getattr(_thread_local, "capture_buf", None)
        if buf is not None:
            return buf.write(data)
        return object.__getattribute__(self, "_real").write(data)

    def flush(self) -> None:
        buf = getattr(_thread_local, "capture_buf", None)
        if buf is not None:
            buf.flush()
        else:
            object.__getattribute__(self, "_real").flush()

    def isatty(self) -> bool:
        return False

    def fileno(self):
        return object.__getattribute__(self, "_real").fileno()

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)


# Install the proxy once at module import time.
if not isinstance(sys.stdout, _ThreadLocalStdoutProxy):
    sys.stdout = _ThreadLocalStdoutProxy(sys.stdout)


@functools.lru_cache(maxsize=None)
def warn_once() -> None:
    """Warn once about the dangers of PythonREPL."""
    logger.warning("Python REPL can execute arbitrary code. Use with caution.")


def python_repl_tool(_globals, _locals, package_names):
    """Create a Python REPL tool definition for function calling."""
    return {
        "type": "function",
        "function": {
            "name": "PythonREPL",
            "description": (
                f"A Python REPL. Use this to execute python code. "
                f"Input should be a valid python command. "
                f"If you want to see the output of a value, you should print it out with `print(...)`. "
                f"You cannot use matplotlib. No plotting is allowed.\n"
                f"Packages you can import: {package_names}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input_code": {
                        "type": "string",
                        "description": "A valid python command."
                    }
                },
                "required": ["input_code"]
            }
        }
    }


# Pre-loaded variable names that should NOT be imported
PRELOADED_VARS = {
    "times_days", "rvs_ms", "sigmas_ms", "np", "baselines", "history",
    "star_mass_sun", "t_ref_days", "stargazer_planet_from_fit", "STARGAZER_SUBMISSION_GUIDE"
}


def _format_traceback(e: Exception, cleaned_command: str) -> str:
    """Format a user-facing traceback for errors in exec'd code."""
    tb = traceback.extract_tb(sys.exc_info()[2])
    user_tb = [frame for frame in tb if frame.filename == "<string>"]
    tb_str = "Error Traceback:\n"
    command_lines = cleaned_command.split("\n")
    for frame in user_tb:
        tb_str += f"  line {frame.lineno}:\n"
        if 0 < frame.lineno <= len(command_lines):
            tb_str += f"    {command_lines[frame.lineno - 1].strip()}\n"
    tb_str += f"{type(e).__name__}: {str(e)}"
    return tb_str


def worker(command: str, namespace: dict, queue: multiprocessing.Queue) -> None:
    """Execute code in a subprocess with stdout redirected to a StringIO.

    This function runs inside a separate *process* (via multiprocessing), so
    replacing sys.stdout here is completely safe — the parent process and all
    other threads are unaffected.
    """
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    try:
        exec(command, namespace)
        sys.stdout = old_stdout
        queue.put(mystdout.getvalue())
    except ModuleNotFoundError as e:
        sys.stdout = old_stdout
        module_name = (
            str(e.name) if hasattr(e, "name")
            else (str(e).split("'")[1] if "'" in str(e) else "")
        )
        if module_name in PRELOADED_VARS:
            queue.put(
                f"ModuleNotFoundError: No module named '{module_name}'\n\n"
                f"HINT: `{module_name}` is a PRE-LOADED VARIABLE, not a module.\n"
                f"Do NOT import it. Just use it directly:\n"
                f"  CORRECT: print({module_name})\n"
                f"  WRONG:   from {module_name} import {module_name}"
            )
        else:
            queue.put(_format_traceback(e, command))
    except Exception as e:
        sys.stdout = old_stdout
        queue.put(_format_traceback(e, command))


def sanitize_input(query: str) -> str:
    """Sanitize input to the Python REPL."""
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    query = re.sub(r"(\s|`)*$", "", query)

    result = []
    i = 0
    in_string = False
    string_char = None

    while i < len(query):
        if not in_string:
            if query[i] in '"\'':
                in_string = True
                string_char = query[i]
                result.append(query[i])
            elif i < len(query) - 1 and query[i:i+2] == '\\n':
                result.append('\n')
                i += 1
            else:
                result.append(query[i])
        else:
            if query[i] == '\\' and i + 1 < len(query):
                result.append(query[i:i+2])
                i += 1
            elif query[i] == string_char:
                in_string = False
                result.append(query[i])
            else:
                result.append(query[i])
        i += 1

    return ''.join(result)


def wrap_last_line_with_print(code: str) -> str:
    """If last line of code is a single word then wrap it in print."""
    lines = code.strip().split('\n')
    last_line = lines[-1].strip()
    if re.match(r'^[^\s,()]+$', last_line):
        lines[-1] = f'print({last_line})'
    return '\n'.join(lines)


def detect_shadowing_callable_conflict(code: str, namespace: dict) -> str | None:
    """Detect assigning to a callable name and then calling it in the same snippet."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    callable_names = {k for k, v in namespace.items() if callable(v)}
    assigned_names = set()
    called_names = set()
    defined_functions = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined_functions.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned_names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigned_names.add(node.target.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            assigned_names.add(node.target.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
            if (
                node.func.id in {"least_squares", "minimize", "root", "curve_fit"}
                and node.args
                and isinstance(node.args[0], ast.Name)
            ):
                callback_name = node.args[0].id
                if callback_name in namespace and not callable(namespace[callback_name]):
                    return (
                        f"Invalid optimizer callback: `{callback_name}` is not callable. "
                        f"It may have been overwritten by a variable. Rename that variable "
                        f"(e.g., `{callback_name}_arr`) or redefine `def {callback_name}(...):`."
                    )

    conflict_names = sorted(
        name for name in assigned_names & called_names
        if name in callable_names or name in defined_functions
    )
    if not conflict_names:
        invalid_calls = sorted(
            name for name in called_names
            if name not in defined_functions and name in namespace and not callable(namespace[name])
        )
        if not invalid_calls:
            return None
        bad_name = invalid_calls[0]
        return (
            f"Invalid call detected: `{bad_name}` is not callable in the current session. "
            f"It may have been overwritten by a variable. Rename the variable (e.g., `{bad_name}_arr`) "
            f"or redefine `def {bad_name}(...):` before calling it."
        )

    conflict = conflict_names[0]
    return (
        f"Name shadowing detected: `{conflict}` is assigned and called in the same code block. "
        f"Use a different variable name like `{conflict}_arr` or `{conflict}_vec`."
    )


def _execute_in_thread(cleaned_code: str, namespace: dict) -> str:
    """Run *cleaned_code* in the current thread with thread-local stdout capture.

    Sets _thread_local.capture_buf so that _ThreadLocalStdoutProxy routes all
    print()/sys.stdout.write() calls in this thread to a private StringIO.
    Other threads continue writing to real stdout unaffected.
    """
    buf = StringIO()
    _thread_local.capture_buf = buf
    try:
        exec(cleaned_code, namespace)
    except ModuleNotFoundError as e:
        module_name = (
            str(e.name) if hasattr(e, "name")
            else (str(e).split("'")[1] if "'" in str(e) else "")
        )
        if module_name in PRELOADED_VARS:
            return (
                f"ModuleNotFoundError: No module named '{module_name}'\n\n"
                f"HINT: `{module_name}` is a PRE-LOADED VARIABLE, not a module.\n"
                f"Do NOT import it. Just use it directly:\n"
                f"  CORRECT: print({module_name})\n"
                f"  WRONG:   from {module_name} import {module_name}"
            )
        return _format_traceback(e, cleaned_code)
    except Exception as e:
        return _format_traceback(e, cleaned_code)
    finally:
        # Always clear the capture buffer so subsequent prints in this thread
        # go back to the real stdout.
        _thread_local.capture_buf = None

    return buf.getvalue()


def execute_python_repl(input_code: str, _globals: dict, _locals: dict, timeout: int = None) -> str:
    """Execute Python code in the REPL."""
    warn_once()
    if 'matplotlib' in input_code:
        return "No plotting is allowed. Code was not executed since it contained 'matplotlib'."

    namespace = {**_globals, **_locals}
    try:
        import sys as _sys
        if "baselines" in namespace and "baselines" not in _sys.modules:
            _sys.modules["baselines"] = namespace["baselines"]
    except Exception:
        pass

    cleaned_code = wrap_last_line_with_print(sanitize_input(input_code))
    conflict_error = detect_shadowing_callable_conflict(cleaned_code, namespace)
    if conflict_error:
        return conflict_error

    if timeout is not None:
        # Subprocess path: stdout redirection inside the child process is safe.
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=worker, args=(cleaned_code, namespace, queue))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            return "Execution timed out"
        result = queue.get()
    else:
        # In-thread path: use thread-local capture to avoid cross-thread pollution.
        result = _execute_in_thread(cleaned_code, namespace)

    _globals.update({k: v for k, v in namespace.items() if k not in _locals})
    _locals.update({k: v for k, v in namespace.items() if k in _locals})

    if len(result) > 5000:
        result = result[:5000] + "...(output truncated)"
    if len(result) == 0:
        result = "No output. You likely forgot to print the result. Please use `print(...)` to see any output."
    return result
