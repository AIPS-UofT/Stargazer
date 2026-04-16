from __future__ import annotations

import ast
import json
import os
import re
import textwrap
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None

try:
    from json_repair import repair_json as _json_repair
except ImportError:  # pragma: no cover - optional helper
    _json_repair = None

from .tools.python_repl_tool import python_repl_tool, execute_python_repl
from .tools.submit_action_tool import submit_action_tool, execute_submit_action
from .openai_utils import create_openai_chat_completion
from ..benchmarks import baselines
from ..limits import DEFAULT_SUBMISSION_MAX_PLANETS

_JSON_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


class MessageLogItem:
    """Container for assistant messages."""

    def __init__(self, text):
        self.content = text


class StepRecord:
    """Record of a single tool execution step."""

    def __init__(self, tool, tool_input, message_log):
        self.tool = tool
        self.tool_input = tool_input
        self.message_log = message_log  # list of MessageLogItem


@dataclass
class TabularAgentConfig:
    """Configuration for the tabular RV agent."""

    model: str = "gpt-5-mini"
    temperature: float = 0.1
    max_tool_calls: int = 12
    max_execution_time: float = 300.0
    max_tokens_per_task: int = 100000
    max_planets: int = DEFAULT_SUBMISSION_MAX_PLANETS
    reasoning_effort: Optional[str] = None
    stream_output: bool = False
    api_timeout_sec: float = 60.0


class TabularRvAgent:
    """
    RV analysis agent following GravityBench's tabular agent architecture.

    The agent uses a ReAct-style loop with:
    - PythonREPL tool for data exploration and analysis
    - submit_action tool for submitting planet hypotheses
    - Budget tracking (tool calls, time, tokens)
    - Multi-provider support (Claude, OpenAI, O-series)
    """

    def __init__(
        self,
        env,
        config: Optional[TabularAgentConfig] = None,
        client: Optional[Any] = None,
        on_step_callback: Optional[Any] = None,
        trace_dir: Optional[Path] = None,
    ):
        self.env = env
        self.config = config or TabularAgentConfig()
        model_name = (self.config.model or "").strip()
        self._provider = "openai_compat"  # "anthropic" or "openai_compat"
        self._api_model = model_name or "gpt-5-mini"
        self._force_responses_api: Optional[bool] = None
        self._use_qwen_json_repair = any(k in self._api_model.lower() for k in ("qwen", "kimi"))
        self.llm = self._initialize_client(client)
        self._on_step_callback = on_step_callback
        self._trace_dir = Path(trace_dir) if trace_dir else None

        # State tracking
        self._current_obs: Optional[Dict[str, Any]] = None
        self._current_info: Optional[Dict[str, Any]] = None
        self._python_globals: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def _flush_trace(self, trace: Dict[str, Any], start_time: float, tool_calls_made: int) -> None:
        """Write current trace state to disk incrementally.

        Called after each tool step so that partial progress survives a hard kill.
        """
        if self._trace_dir is None:
            return
        try:
            self._trace_dir.mkdir(parents=True, exist_ok=True)
            snapshot = {
                "input": trace.get("input"),
                "intermediate_steps": [
                    {
                        "tool_info": {
                            "tool": step[0].tool,
                            "tool_input": step[0].tool_input,
                        },
                        "tool_output": step[1],
                    }
                    for step in trace.get("intermediate_steps", [])
                ],
                "output": trace.get("output"),
                "error_message": trace.get("error_message"),
                "history": self._history,
                "input_tokens_used": self._total_input_tokens,
                "output_tokens_used": self._total_output_tokens,
                "tool_calls_made": tool_calls_made,
                "elapsed_seconds": round(time.time() - start_time, 1),
                "stop_reason": trace.get("stop_reason") or "in_progress",
            }
            if self._on_step_callback and hasattr(self._on_step_callback, "get_mentor_stats"):
                snapshot["mentor_stats"] = self._on_step_callback.get_mentor_stats()
            out_path = self._trace_dir / "trace_incremental.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, default=str)
        except Exception:
            pass  # incremental flush is best-effort

    def _initialize_client(self, client: Optional[Any]) -> Any:
        """Initialize API client based on model name."""
        if client is not None:
            return client

        model_name = self.config.model.strip()
        model_lower = model_name.lower()

        # Explicit OpenRouter route: model string like "openrouter/<provider/model>"
        if model_lower.startswith("openrouter/"):
            if openai is None:
                raise ImportError(
                    "openai package required for OpenRouter models. "
                    "Install with: pip install openai"
                )
            routed_model = model_name.split("/", 1)[1].strip()
            if not routed_model:
                raise ValueError(
                    "OpenRouter model must be provided as 'openrouter/<provider/model>'."
                )
            api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Missing OPENROUTER_API_KEY (or OPENAI_API_KEY as fallback) for OpenRouter."
                )
            base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            headers: Dict[str, str] = {}
            referer = os.environ.get("OPENROUTER_HTTP_REFERER")
            app_name = os.environ.get("OPENROUTER_APP_NAME")
            if referer:
                headers["HTTP-Referer"] = referer
            if app_name:
                headers["X-Title"] = app_name

            self._provider = "openai_compat"
            self._api_model = routed_model
            # OpenRouter provides OpenAI-compatible chat.completions; disable Responses API routing.
            self._force_responses_api = False
            return openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.config.api_timeout_sec,
                default_headers=(headers or None),
            )

        if "claude" in model_lower:
            if anthropic is None:
                raise ImportError(
                    "anthropic package required for Claude models. "
                    "Install with: pip install anthropic"
                )
            self._provider = "anthropic"
            self._api_model = model_name
            self._force_responses_api = False
            return anthropic.Anthropic(timeout=self.config.api_timeout_sec)
        elif "intern" in model_lower:
            if openai is None:
                raise ImportError(
                    "openai package required for InternLM models. "
                    "Install with: pip install openai"
                )
            api_key = os.environ.get("INTERNLM_API_KEY", "")
            self._provider = "openai_compat"
            self._api_model = model_name
            self._force_responses_api = False
            return openai.OpenAI(
                api_key=api_key,
                base_url="https://chat.intern-ai.org.cn/api/v1/",
                timeout=self.config.api_timeout_sec,
            )
        elif "gpt" in model_lower or model_lower.startswith("o"):
            if openai is None:
                raise ImportError(
                    "openai package required for OpenAI models. "
                    "Install with: pip install openai"
                )
            self._provider = "openai_compat"
            self._api_model = model_name
            self._force_responses_api = None
            return openai.OpenAI(timeout=self.config.api_timeout_sec)
        else:
            # Fallback: if OpenRouter key is present, allow arbitrary model names via OpenRouter.
            if openai is not None and os.environ.get("OPENROUTER_API_KEY"):
                base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
                headers: Dict[str, str] = {}
                referer = os.environ.get("OPENROUTER_HTTP_REFERER")
                app_name = os.environ.get("OPENROUTER_APP_NAME")
                if referer:
                    headers["HTTP-Referer"] = referer
                if app_name:
                    headers["X-Title"] = app_name
                self._provider = "openai_compat"
                self._api_model = model_name
                self._force_responses_api = False
                return openai.OpenAI(
                    api_key=os.environ["OPENROUTER_API_KEY"],
                    base_url=base_url,
                    timeout=self.config.api_timeout_sec,
                    default_headers=(headers or None),
                )

            raise ValueError(
                f"Model {self.config.model} not recognized. "
                f"Use 'claude-*' for Anthropic, 'openrouter/<provider/model>' for OpenRouter, "
                f"'intern*' for InternLM, or 'gpt-*'/'o*' for OpenAI."
            )

    def _convert_tools_for_anthropic(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI tool format to Anthropic format."""
        return [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in tools
            if t["type"] == "function"
        ]

    def _build_system_prompt(self) -> str:
        """Build the system prompt with task context."""
        obs = self._current_obs
        times = np.asarray(obs["times_days"], dtype=float)
        rvs = np.asarray(obs["rvs_ms"], dtype=float)
        sigmas = np.asarray(obs["sigmas_ms"], dtype=float)
        span = float(times[-1] - times[0]) if len(times) > 1 else 0.0
        median_sigma = float(np.median(sigmas))

        # Check for task-specific description and hints
        meta = obs.get("meta", {})
        task_description = meta.get("task_description", "")
        hints = meta.get("hints", {})
        reference = meta.get("reference", "")

        # Build task context section if available
        task_context_section = ""
        if task_description:
            task_context_section = f"\n{task_description}\n"
        if hints:
            hints_text = "\n".join(f"- {k}: {v}" for k, v in hints.items())
            task_context_section += f"\n### Hints\n{hints_text}\n"
        if reference:
            task_context_section += f"\n### Reference\n{reference}\n"

        prompt = textwrap.dedent(
            f"""
            You are an expert RV data analyst tasked with detecting exoplanets from radial velocity measurements.
            {task_context_section}
            ### Dataset Overview
            - Number of observations: {len(times)}
            - Time span: {span:.1f} days
            - RV range: {rvs.min():+.2f} to {rvs.max():+.2f} m/s
            - Median uncertainty: {median_sigma:.2f} m/s

            ### Your Task
            Analyze the RV data to identify planetary signals. Use the PythonREPL tool to:
            - Compute periodograms (Lomb-Scargle or other methods)
            - Test baseline models (available via `baselines` module)
            - Fit Keplerian orbital models (NOT simple sinusoids)
            - Analyze residuals and trigger optimization when needed

            ### CRITICAL: Keplerian Model Parameters
            When fitting Keplerian orbits, you MUST fit ALL of these parameters:
            - **P**: Period (days)
            - **K**: RV semi-amplitude (m/s)
            - **e**: Eccentricity (0 to 0.8)
            - **omega**: Argument of periastron (radians, 0 to 2π) - CRITICAL FOR ECCENTRIC ORBITS!
            - **M0**: Mean anomaly at reference time (radians)
            - **gamma**: Systemic velocity offset (m/s)

            For eccentric orbits (e > 0.1), the omega parameter significantly affects the RV curve shape.
            Always include omega in your fit AND in your submission!

            ### Submission Format
            Use Stargazer-native planet fields for highest reliability:
            - `P_days`, `m_sin_i_mjup`, `e`, `omega_rad`, `l_rad`
            - `l_rad` is mean longitude at reference epoch `t_ref = times_days[0]`
            - If your fit gives mean anomaly `M0` at `t_ref`, convert with: `l_rad = (Omega_rad + omega_rad + M0) % (2π)`

            When ready to submit, call submit_action with your BEST fitted parameters:
            ```python
            {{
                'planets': [{{
                    'P_days': P,
                    'm_sin_i_mjup': m_sin_i,
                    'e': e,
                    'omega_rad': omega,
                    'l_rad': l_rad,
                    'inc_rad': inc,      # Optional, REBOUND geometry
                    'Omega_rad': Omega,  # Optional, REBOUND geometry
                }}],
                'rv_offset_ms': gamma,          # Systemic velocity
                'noise_jitter_ms': 0.5,         # Optional jitter term
            }}
            ```

            Use helper function `stargazer_planet_from_fit(...)` in PythonREPL to convert
            `(P, K, e, omega, M0)` into a correct Stargazer planet dict.

            ### Response format for every turn
            1) Findings: concise hypothesis plus key numbers (candidate periods/powers/RMS).
            2) Plan/Next: 1–3 short bullets of what you'll do next.
            3) Code: one fenced code block with what you will run now (only if calling PythonREPL).
            4) Results: printed outputs interpreted; if ready, include submit_action parameters.
            - Keep prose and code separate; do not mix explanations inside code blocks.
            - Always print key metrics from code; avoid silent computations.

            ### Available Tools
            1. **PythonREPL**: Execute Python code for analysis
               - Pre-loaded variables (DO NOT import, just use directly):
                 `times_days`, `rvs_ms`, `sigmas_ms`, `np`, `baselines`, `history`,
                 `star_mass_sun`, `t_ref_days`, `stargazer_planet_from_fit`, `STARGAZER_SUBMISSION_GUIDE`
               - Example: `print(times_days.max() - times_days.min())`  # Correct
               - WRONG: `from times_days import times_days`  # Do NOT do this!
               - Always use print() to see outputs
               - No plotting allowed

            2. **submit_action**: Submit planet hypotheses
               - Max {self.config.max_planets} planets
               - Period must be > 0.5 days
               - Eccentricity: 0 to 0.8
               - Submission mode: {self.env.submission_mode}

            ### Budget Constraints
            - Max tool calls: {self.config.max_tool_calls}
            - Max execution time: {self.config.max_execution_time}s
            - You can submit up to {self.env.max_steps} times

            ### Mandatory Step 0: Read Protocol Guide First
            Before any fitting/submission, read `STARGAZER_SUBMISSION_GUIDE` in PythonREPL and set:
            `_protocol_guide_ack = True`.
            You are NOT allowed to call `submit_action` until this is done.

            ### Strategy (FOLLOW THIS ORDER)

            **Step 1: Periodogram Analysis**
            - Compute Lomb-Scargle periodogram
            - Identify strongest peak(s) and their periods

            **Step 2: Linear Sine Baseline (MANDATORY MODEL GATING)**
            - Before any Keplerian optimization, you MUST run a linear/sinusoidal baseline first.
            - Use `baselines.baseline_one_sine(observation)` (or equivalent linear sine fit).
            - Print at least: candidate period, baseline RMS, and RMS/median_sigma.

            **Step 3: Model Gating Decision (MANDATORY)**
            - Decide whether Kepler is needed based on baseline diagnostics.
            - If baseline RMS is already close to noise (RMS/median_sigma <= 1.5), prefer direct submission/refinement.
            - If baseline RMS is not close to noise, escalate to full Keplerian fitting.
            - Explicitly state: `Gate decision: Kepler=YES/NO` before running Kepler code.

            **Step 4: Keplerian Fitting (ONLY IF GATE=YES)**
            - Fit a FULL 6-parameter Keplerian: P, K, e, omega, M0, gamma
            - Use scipy.optimize.least_squares with bounds
            - For high eccentricity (e > 0.3), try multiple omega starting values
            - Use multi-start optimization to avoid local minima

            **Step 5: Check Fit Quality**
            - Compute residual RMS after fitting
            - Good fit: RMS ≈ {median_sigma:.2f} m/s (close to measurement uncertainty)
            - Bad fit: RMS >> {median_sigma:.2f} m/s → keep optimizing

            **Step 6: Submit ONLY After Convergence**
            - DO NOT submit until RMS is close to noise level
            - Include ALL fitted parameters in submission, especially omega_rad!
            - Double-check: did you include omega_rad in your submission?

            ### Common Mistakes to AVOID
            1. Jumping to Kepler before LS + linear-sine gating
            2. Submitting early with poor fit (high RMS)
            3. Forgetting omega_rad in submission (it will default to 0!)
            4. Using wrong phase convention (`l_rad` is mean longitude, not raw phase offset)
            5. Not doing multi-start optimization for eccentric orbits
            6. Reusing function names as variables in Python (e.g., `residuals = ...` after `def residuals(...)`).
               - If you define a function `residuals`, keep it callable.
               - Use names like `residual_vec`, `fit_residuals`, `model_rv_arr` for arrays.

            A successful fit should achieve residual RMS ≈ {median_sigma:.2f} m/s.
            If your RMS is much larger, your fit has NOT converged - keep optimizing!

            ### Mentor Guidance
            You may receive `[Mentor guidance]` messages from an expert reviewer.
            Treat this advice as high-priority — follow it before continuing your analysis.
            """
        ).strip()
        return prompt

    def _load_protocol_guide(self) -> str:
        """Load protocol guide text from repository file with a safe fallback."""
        guide_path = Path(__file__).resolve().parents[2] / "STARGAZER_SUBMISSION_GUIDE.md"
        fallback = (
            "Stargazer Submission Guide\\n"
            "1) Preferred fields: P_days, m_sin_i_mjup, e, omega_rad, l_rad\\n"
            "2) Reference epoch: t_ref = times_days[0]\\n"
            "3) Convert M0 to l_rad via l_rad = (Omega_rad + omega_rad + M0) mod 2pi\\n"
            "4) Avoid mixing phase aliases; if using l_rad, treat it as canonical\\n"
            "5) Before submit: verify converted action rv_model residual RMS is near sigma\\n"
        )
        try:
            return guide_path.read_text(encoding="utf-8")
        except Exception:
            return fallback

    def run(self, verbose: bool = False) -> Dict[str, Any]:
        # Reset environment and state
        self._current_obs, self._current_info = self.env.reset()
        self._history.clear()
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        # Initialize Python REPL environment
        star_mass_sun = float(self._current_obs.get("meta", {}).get("star_mass_sun", 1.0))
        t_ref = float(np.asarray(self._current_obs["times_days"], dtype=float)[0])
        protocol_guide = self._load_protocol_guide()

        def stargazer_planet_from_fit(
            P_days: float,
            K_ms: float,
            e: float = 0.0,
            omega_rad: float = 0.0,
            M0_rad: float = 0.0,
            inc_rad: float = float(np.pi / 2.0),
            Omega_rad: float = 0.0,
            m_sin_i_mjup: Optional[float] = None,
        ) -> Dict[str, float]:
            """Convert fitted Keplerian params into canonical Stargazer planet fields."""
            P_days_f = float(P_days)
            K_ms_f = float(max(0.0, K_ms))
            e_f = float(np.clip(e, 0.0, 0.8))
            omega_f = float(omega_rad % (2.0 * np.pi))
            M0_f = float(M0_rad % (2.0 * np.pi))
            inc_f = float(np.clip(inc_rad, 0.0, np.pi))
            Omega_f = float(Omega_rad % (2.0 * np.pi))
            l_rad = float((Omega_f + omega_f + M0_f) % (2.0 * np.pi))

            if m_sin_i_mjup is None:
                P_years = P_days_f / 365.25
                denom = (
                    28.4329
                    * (star_mass_sun ** (-2.0 / 3.0))
                    * (P_years ** (-1.0 / 3.0))
                    / np.sqrt(max(1e-12, 1.0 - e_f * e_f))
                )
                msi = float(np.clip(K_ms_f / denom, 1e-3, 30.0))
            else:
                msi = float(np.clip(m_sin_i_mjup, 1e-3, 30.0))

            return {
                "P_days": P_days_f,
                "m_sin_i_mjup": msi,
                "e": e_f,
                "inc_rad": inc_f,
                "Omega_rad": Omega_f,
                "omega_rad": omega_f,
                "l_rad": l_rad,
            }

        self._python_globals = {
            "np": np,
            "times_days": np.asarray(self._current_obs["times_days"], dtype=float),
            "rvs_ms": np.asarray(self._current_obs["rvs_ms"], dtype=float),
            "sigmas_ms": np.asarray(self._current_obs["sigmas_ms"], dtype=float),
            "baselines": baselines,
            "history": self._history,
            "star_mass_sun": star_mass_sun,
            "t_ref_days": t_ref,
            "stargazer_planet_from_fit": stargazer_planet_from_fit,
            "STARGAZER_SUBMISSION_GUIDE": protocol_guide,
            "_protocol_guide_ack": False,
        }

        # Build tools
        package_names = (
            "np, times_days, rvs_ms, sigmas_ms, baselines, history, "
            "star_mass_sun, t_ref_days, stargazer_planet_from_fit, STARGAZER_SUBMISSION_GUIDE"
        )
        tools = [
            python_repl_tool(
                _globals=self._python_globals,
                _locals=self._python_globals,
                package_names=package_names,
            ),
            submit_action_tool(
                max_planets=self.config.max_planets,
                submission_mode=self.env.submission_mode,
            ),
        ]

        # Initialize message history
        system_prompt = self._build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Begin your analysis."},
        ]

        # Trace for logging
        trace = {
            "input": system_prompt,
            "intermediate_steps": [],
            "output": None,
            "error_message": None,
        }

        if verbose:
            print(f"[System] {system_prompt[:200]}...")

        start_time = time.time()
        tool_calls_made = 0
        last_done = False
        stop_reason: Optional[str] = None
        force_submit_now = False
        last_text_norm: Optional[str] = None
        repeat_text_count = 0
        no_tool_turns = 0

        def _update_submit_gate_from_text(text: str) -> bool:
            """Gate rule: if Kepler=YES and Best_RMS_over_med_sigma < 1.1, force next action to submit."""
            if not text:
                return False
            if re.search(r"Kepler\s*=\s*YES", text, flags=re.IGNORECASE) is None:
                return False
            m = re.search(
                r"Best_RMS_over_med_sigma\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                text,
            )
            if m is None:
                return False
            try:
                ratio = float(m.group(1))
            except ValueError:
                return False
            return ratio < 1.1

        try:
            while (
                tool_calls_made < self.config.max_tool_calls
                and time.time() - start_time < self.config.max_execution_time - 2
                and not last_done
            ):
                if verbose:
                    print(
                        f"\n[Progress] Tool calls: {tool_calls_made}/{self.config.max_tool_calls}, "
                        f"Time: {time.time() - start_time:.1f}/{self.config.max_execution_time}s"
                    )

                # Get LLM response
                try:
                    call_start = time.time()
                    if not verbose:
                        print(
                            f"[API] Calling {self.config.model} (step {len(trace['intermediate_steps']) + 1})...",
                            flush=True,
                        )
                    trimmed_messages = self._trim_message_history(messages)
                    if verbose:
                        # Avoid dumping the full conversation on every turn.
                        print(
                            f"[Debug] Calling model with {len(trimmed_messages)} messages "
                            f"(trimmed from {len(messages)}); "
                            f"last roles={[m.get('role') for m in trimmed_messages[-4:]]}"
                        )
                    if self._provider == "anthropic":
                        response = self._call_anthropic(trimmed_messages, tools)
                    else:
                        response = self._call_openai(trimmed_messages, tools, verbose=verbose)
                    if not verbose:
                        print(
                            f"[API] Response received in {time.time() - call_start:.1f}s",
                            flush=True,
                        )
                except Exception as api_error:
                    error_msg = f"API Error: {type(api_error).__name__} - {api_error}"
                    print(f"[ERROR] {error_msg}")
                    trace["error_message"] = error_msg + f"\n{traceback.format_exc()}"
                    raise

                content = response["content"]
                tool_calls = response.get("tool_calls")

                if verbose and content.strip() and not response.get("streamed_text"):
                    print(f"[Assistant] {content}")

                # Process tool calls
                if tool_calls:
                    repeat_text_count = 0
                    no_tool_turns = 0
                    last_text_norm = None
                    tool_calls_made += len(tool_calls)

                    # Add assistant message with tool calls
                    if self._provider == "anthropic":
                        assistant_content = []
                        if content.strip():
                            assistant_content.append({"type": "text", "text": content})
                        for tc in tool_calls:
                            assistant_content.append(
                                {
                                    "type": "tool_use",
                                    "id": tc["id"],
                                    "name": tc["name"],
                                    "input": tc["input"],
                                }
                            )
                        messages.append({"role": "assistant", "content": assistant_content})
                    else:
                        formatted_tool_calls = [
                            self._format_tool_call_for_openai(tc) for tc in tool_calls
                        ]
                        messages.append(
                            {
                                "role": "assistant",
                                "content": content,
                                "tool_calls": formatted_tool_calls,
                            }
                        )

                    # Execute each tool
                    for i, tool_call in enumerate(tool_calls):
                        tool_name = tool_call["name"]
                        tool_args = tool_call["input"]

                        if verbose:
                            print(f"\n[Tool] {tool_name}")
                            print(f"[Args] {json.dumps(tool_args, indent=2)}")

                        msg_log = []
                        if i == 0 and content.strip():
                            msg_log.append(MessageLogItem(content))

                        # Enforce submit gate: do not allow non-submit tools once gate is active.
                        if force_submit_now and tool_name != "submit_action":
                            result = (
                                "Policy gate active: Best_RMS_over_med_sigma < 1.1 with Kepler=YES. "
                                "Your next step MUST be submit_action now; skip summaries and extra analysis."
                            )
                            if verbose:
                                print(f"[Output] {result[:500]}")
                            if self._provider == "anthropic":
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "tool_result",
                                                "tool_use_id": tool_call["id"],
                                                "content": str(result),
                                            }
                                        ],
                                    }
                                )
                            else:
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call["id"],
                                        "content": str(result),
                                    }
                                )
                            trace["intermediate_steps"].append(
                                [
                                    StepRecord(
                                        tool=tool_name, tool_input=tool_args, message_log=msg_log
                                    ),
                                    str(result),
                                ]
                            )
                            continue

                        # Execute tool
                        if tool_name == "PythonREPL":
                            result = execute_python_repl(
                                tool_args["input_code"],
                                self._python_globals,
                                self._python_globals,
                            )
                            if _update_submit_gate_from_text(str(result)):
                                force_submit_now = True
                        elif tool_name == "submit_action":
                            if not bool(self._python_globals.get("_protocol_guide_ack", False)):
                                result = (
                                    "Submission blocked: protocol guide not acknowledged yet. "
                                    "Run PythonREPL to read `STARGAZER_SUBMISSION_GUIDE`, then set "
                                    "`_protocol_guide_ack = True` before calling submit_action."
                                )
                                obs_next = self._current_obs
                                reward = 0.0
                                done = False
                                info = {}
                                self._current_obs = obs_next
                                last_done = done
                            elif self._on_step_callback:
                                # Hook: pre_submit — let mentor review before execution
                                mentor_guidance = self._on_step_callback(
                                    event="pre_submit",
                                    tool_name=tool_name,
                                    tool_args=tool_args,
                                    tool_result=None,
                                    messages=messages,
                                    tool_calls_made=tool_calls_made,
                                    history=self._history,
                                )
                                if mentor_guidance:
                                    # Mentor blocked the submission — return guidance as tool result
                                    result = (
                                        f"[Mentor review]: {mentor_guidance}\n"
                                        "Please revise your submission based on the mentor's feedback."
                                    )
                                    if verbose:
                                        print(f"[Mentor] Blocked submission: {mentor_guidance[:200]}")
                                else:
                                    # Mentor approved — proceed normally
                                    result, obs_next, reward, done, info = execute_submit_action(
                                        tool_args,
                                        self.env,
                                        self._current_obs,
                                        self.config.max_planets,
                                    )
                                    self._current_obs = obs_next
                                    last_done = done
                                    force_submit_now = False
                                    self._history.append(
                                        {
                                            "step": len(self._history) + 1,
                                            "reward": float(reward),
                                            "done": bool(done),
                                            "success": bool(info.get("success", False)),
                                            "success_details": info.get("success_details", {}),
                                            "metrics": info.get("metrics", {}),
                                        }
                                    )
                                    self._python_globals["history"] = self._history

                                    # Hook: post_submit — let mentor diagnose failures
                                    if not info.get("success", False) and self._on_step_callback:
                                        post_guidance = self._on_step_callback(
                                            event="post_submit",
                                            tool_name=tool_name,
                                            tool_args=tool_args,
                                            tool_result=str(result),
                                            messages=messages,
                                            tool_calls_made=tool_calls_made,
                                            history=self._history,
                                            last_reward=float(reward),
                                        )
                                        if post_guidance:
                                            result = str(result) + f"\n\n[Mentor guidance]: {post_guidance}"
                            else:
                                result, obs_next, reward, done, info = execute_submit_action(
                                    tool_args,
                                    self.env,
                                    self._current_obs,
                                    self.config.max_planets,
                                )
                                self._current_obs = obs_next
                                last_done = done
                                force_submit_now = False

                                # Track submission in history
                                self._history.append(
                                    {
                                        "step": len(self._history) + 1,
                                        "reward": float(reward),
                                        "done": bool(done),
                                        "success": bool(info.get("success", False)),
                                        "success_details": info.get("success_details", {}),
                                        "metrics": info.get("metrics", {}),
                                    }
                                )
                                self._python_globals["history"] = self._history
                        else:
                            result = f"Unknown tool: {tool_name}"

                        if verbose:
                            print(f"[Output] {result[:500]}")

                        # Add tool result to messages
                        if self._provider == "anthropic":
                            messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": tool_call["id"],
                                            "content": str(result),
                                        }
                                    ],
                                }
                            )
                        else:
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "content": str(result),
                                }
                            )

                        # Record step in trace
                        trace["intermediate_steps"].append(
                            [
                                StepRecord(
                                    tool=tool_name, tool_input=tool_args, message_log=msg_log
                                ),
                                str(result),
                            ]
                        )
                        self._flush_trace(trace, start_time, tool_calls_made)

                        # Hook: post_tool — let mentor review after tool execution
                        if (
                            self._on_step_callback
                            and tool_name == "PythonREPL"
                            and not last_done
                        ):
                            post_tool_guidance = self._on_step_callback(
                                event="post_tool",
                                tool_name=tool_name,
                                tool_args=tool_args,
                                tool_result=str(result),
                                messages=messages,
                                tool_calls_made=tool_calls_made,
                                history=self._history,
                            )
                            if post_tool_guidance:
                                if verbose:
                                    print(f"[Mentor] {post_tool_guidance[:200]}")
                                messages.append(
                                    {"role": "user", "content": f"[Mentor guidance]: {post_tool_guidance}"}
                                )

                        if last_done:
                            if verbose:
                                print("\n[Done] Environment episode complete.")
                            break

                else:
                    if force_submit_now:
                        # Hard nudge: do not allow narrative-only turn while submit gate is active.
                        reminders = (
                            "Policy gate active: Best_RMS_over_med_sigma < 1.1 and Kepler=YES. "
                            "Call submit_action immediately."
                        )
                        if self._provider == "anthropic":
                            messages.append({"role": "user", "content": reminders})
                        else:
                            messages.append({"role": "user", "content": reminders})
                        continue

                    # No tool calls - just add message
                    if content.strip():
                        no_tool_turns += 1
                        norm = re.sub(r"\s+", " ", content.strip().lower())
                        if norm == last_text_norm:
                            repeat_text_count += 1
                        else:
                            repeat_text_count = 1
                            last_text_norm = norm
                        if repeat_text_count >= 3:
                            stop_reason = "loop_detected"
                            trace["error_message"] = (
                                "Loop detected: repeated assistant text without tool calls."
                            )
                            trace["intermediate_steps"].append(
                                [
                                    StepRecord(
                                        tool="Agent Error",
                                        tool_input="",
                                        message_log=[
                                            MessageLogItem(
                                                "Loop detected: repeated assistant text without tool calls."
                                            )
                                        ],
                                    ),
                                    "Loop detected",
                                ]
                            )
                            break

                        # Force tool usage after too many text-only turns
                        MAX_NO_TOOL_TURNS = 2
                        if no_tool_turns >= MAX_NO_TOOL_TURNS:
                            # Hook: stuck — let mentor provide direction
                            if self._on_step_callback:
                                stuck_guidance = self._on_step_callback(
                                    event="stuck",
                                    tool_name=None,
                                    tool_args=None,
                                    tool_result=None,
                                    messages=messages,
                                    tool_calls_made=tool_calls_made,
                                    history=self._history,
                                )
                                if stuck_guidance:
                                    nudge_msg = f"[Mentor guidance]: {stuck_guidance}"
                                else:
                                    nudge_msg = (
                                        "You must call a tool now. Do NOT ask for user input - "
                                        "you are running autonomously. Either call PythonREPL to "
                                        "continue analysis or submit_action to submit your results."
                                    )
                            else:
                                nudge_msg = (
                                    "You must call a tool now. Do NOT ask for user input - "
                                    "you are running autonomously. Either call PythonREPL to "
                                    "continue analysis or submit_action to submit your results."
                                )
                            messages.append({"role": "user", "content": nudge_msg})
                            no_tool_turns = 0
                            if verbose:
                                print(f"[Nudge] Forced tool usage after {MAX_NO_TOOL_TURNS} text-only turns")
                            continue

                        trace["intermediate_steps"].append(
                            [
                                StepRecord(
                                    tool="Assistant Text",
                                    tool_input="",
                                    message_log=[MessageLogItem(content)],
                                ),
                                "",
                            ]
                        )
                    messages.append({"role": "assistant", "content": content})

                # Check token budget
                if self._total_input_tokens + self._total_output_tokens > (
                    self.config.max_tokens_per_task - 1000
                ):
                    print(
                        f"[Warning] Approaching token limit: "
                        f"{self._total_input_tokens + self._total_output_tokens}/{self.config.max_tokens_per_task}"
                    )
                    if not last_done:
                        stop_reason = "token_budget"
                    break

        except Exception as e:
            error_name = type(e).__name__
            error_description = str(e)
            error_traceback = traceback.format_exc()
            print(f"\n[ERROR] {error_name}: {error_description}")
            if verbose:
                print(f"Traceback:\n{error_traceback}")

            trace["intermediate_steps"].append(
                [
                    StepRecord(
                        tool="Agent Error",
                        tool_input="",
                        message_log=[
                            MessageLogItem(f"Error: {error_name} - {error_description}")
                        ],
                    ),
                    f"Agent encountered error: {error_name}",
                ]
            )
            trace["error_message"] = (
                f"{error_name} - {error_description}\n{error_traceback}"
            )

        finally:
            # Finalize trace
            if self._history:
                trace["output"] = self._history[-1]
            trace["input_tokens_used"] = self._total_input_tokens
            trace["output_tokens_used"] = self._total_output_tokens
            trace["history"] = self._history
            trace["messages"] = messages  # full LLM conversation for inspection
            if stop_reason is None:
                elapsed = time.time() - start_time
                if last_done:
                    stop_reason = "env_done"
                elif tool_calls_made >= self.config.max_tool_calls:
                    stop_reason = "max_tool_calls"
                elif elapsed >= self.config.max_execution_time - 2:
                    stop_reason = "max_execution_time"
                else:
                    stop_reason = "unknown"
            trace["stop_reason"] = stop_reason

            # Include mentor usage stats if a mentor callback was used
            if self._on_step_callback and hasattr(self._on_step_callback, "get_mentor_stats"):
                trace["mentor_stats"] = self._on_step_callback.get_mentor_stats()

            if verbose:
                print(f"\n[Summary] Total tokens: {self._total_input_tokens + self._total_output_tokens}")
                print(f"[Summary] Tool calls: {tool_calls_made}")
                print(f"[Summary] Submissions: {len(self._history)}")
                print(f"[Summary] Stop reason: {stop_reason}")

            # Final incremental flush (with actual stop_reason)
            self._flush_trace(trace, start_time, tool_calls_made)

            return trace

    def _trim_message_history(
        self,
        messages: List[Dict],
        keep_recent_pairs: int = 4,
        truncate_to: int = 300,
    ) -> List[Dict]:
        """Return a trimmed copy of the message list for the API call.

        Strategy:
        - Always keep messages[0] (system) and messages[1] (initial user).
        - Keep the last `keep_recent_pairs * 2` messages in full.
        - For messages in between, truncate tool result content to `truncate_to` chars.
        - The original `messages` list is never mutated.
        """
        # Nothing to trim if history is short
        cutoff = 2 + keep_recent_pairs * 2
        if len(messages) <= cutoff:
            return messages

        tail_start = len(messages) - keep_recent_pairs * 2
        result: List[Dict] = []

        for i, msg in enumerate(messages):
            if i < 2 or i >= tail_start:
                result.append(msg)
                continue

            role = msg.get("role", "")
            if role == "tool":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > truncate_to:
                    msg = dict(msg)
                    msg["content"] = content[:truncate_to] + "...[old output truncated]"
            elif role == "user" and isinstance(msg.get("content"), list):
                # Claude-style: tool results live in user messages as tool_result blocks
                new_blocks = []
                changed = False
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        c = block.get("content", "")
                        if isinstance(c, str) and len(c) > truncate_to:
                            block = dict(block)
                            block["content"] = c[:truncate_to] + "...[old output truncated]"
                            changed = True
                    new_blocks.append(block)
                if changed:
                    msg = dict(msg)
                    msg["content"] = new_blocks

            result.append(msg)

        return result

    def _call_anthropic(
        self, messages: List[Dict], tools: List[Dict]
    ) -> Dict[str, Any]:
        """Call Anthropic API and normalize response."""
        # Extract system message
        system_msg = None
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                anthropic_messages.append(msg)

        # Wrap system prompt as a cacheable block
        system_payload = [
            {"type": "text", "text": system_msg, "cache_control": {"type": "ephemeral"}}
        ]

        # Cache the tool definitions (mark the last tool)
        anthropic_tools = self._convert_tools_for_anthropic(tools)
        if anthropic_tools:
            anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}

        response = self.llm.messages.create(
            model=self.config.model,
            max_tokens=20000,
            system=system_payload,
            messages=anthropic_messages,
            tools=anthropic_tools,
            temperature=self.config.temperature,
        )

        self._total_input_tokens += getattr(response.usage, "input_tokens", 0)
        self._total_input_tokens += getattr(response.usage, "cache_creation_input_tokens", 0)
        self._total_input_tokens += getattr(response.usage, "cache_read_input_tokens", 0)
        self._total_output_tokens += getattr(response.usage, "output_tokens", 0)

        # Convert to normalized format
        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {"id": block.id, "name": block.name, "input": block.input}
                )

        return {"content": content, "tool_calls": tool_calls or None}

    def _call_openai(
        self, messages: List[Dict], tools: List[Dict], verbose: bool = False
    ) -> Dict[str, Any]:
        """Call OpenAI API and normalize response."""
        reasoning_effort = (
            self.config.reasoning_effort
            if self.config.model.lower().startswith("o")
            else None
        )
        use_stream = (
            self.config.stream_output
            and verbose
            and not self.config.model.lower().startswith("gpt-5")
        )
        streamed_any_text = False

        def _print_delta(delta: str) -> None:
            nonlocal streamed_any_text
            if not streamed_any_text:
                print("[Assistant stream] ", end="", flush=True)
            print(delta, end="", flush=True)
            streamed_any_text = True

        response = create_openai_chat_completion(
            self.llm,
            model=self._api_model,
            messages=messages,
            tools=tools,
            temperature=(
                None
                if self.config.model.lower().startswith("o")
                else self.config.temperature
            ),
            reasoning_effort=reasoning_effort,
            stream=use_stream,
            on_text_delta=_print_delta if use_stream else None,
            use_responses_api=self._force_responses_api,
        )
        if use_stream and streamed_any_text:
            print()

        prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        self._total_input_tokens += int(prompt_tokens)
        self._total_output_tokens += int(completion_tokens)

        assistant_message = response.choices[0].message
        content = assistant_message.content or ""
        tool_calls_raw = assistant_message.tool_calls

        # Convert to normalized format
        tool_calls = None
        if tool_calls_raw:
            tool_calls = []
            for tc in tool_calls_raw:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": self._parse_tool_call_input(tc.function.arguments),
                    }
                )

        return {
            "content": content,
            "tool_calls": tool_calls,
            "streamed_text": use_stream and streamed_any_text,
        }

    def _format_tool_call_for_openai(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Convert normalized tool call dict into OpenAI-compatible payload."""
        return {
            "id": tool_call.get("id"),
            "type": "function",
            "function": {
                "name": tool_call.get("name"),
                "arguments": json.dumps(tool_call.get("input", {})),
            },
        }

    def _parse_tool_call_input(self, arguments: Any) -> Dict[str, Any]:
        """Parse tool call arguments, applying Qwen-specific repair when needed."""
        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str):
            return arguments or {}
        if not arguments.strip():
            return {}
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            repaired = self._attempt_qwen_json_repair(arguments)
            if repaired is not None:
                return repaired
            raise

    def _attempt_qwen_json_repair(self, raw: str) -> Optional[Dict[str, Any]]:
        """Best-effort JSON repair for Qwen outputs only."""
        if not self._use_qwen_json_repair:
            return None

        snippets: List[str] = []
        for match in _JSON_CODE_FENCE_RE.finditer(raw):
            chunk = match.group(1).strip()
            if chunk:
                snippets.append(chunk)

        stripped = raw.strip()
        if stripped:
            snippets.append(stripped)

        def _extract_segment(text: str, open_ch: str, close_ch: str) -> Optional[str]:
            start = text.find(open_ch)
            end = text.rfind(close_ch)
            if start == -1 or end == -1 or end <= start:
                return None
            return text[start : end + 1]

        brace_segment = _extract_segment(stripped, "{", "}")
        bracket_segment = _extract_segment(stripped, "[", "]")
        if brace_segment:
            snippets.append(brace_segment)
        if bracket_segment:
            snippets.append(bracket_segment)

        seen: set[str] = set()
        for snippet in snippets:
            candidate = snippet.strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)

            # First try optional third-party repair library
            if _json_repair is not None:
                try:
                    repaired = _json_repair(candidate)
                    data = json.loads(repaired)
                    if isinstance(data, dict):
                        return data
                except Exception:
                    pass

            # Remove trailing commas before closing braces/brackets
            normalized = _TRAILING_COMMA_RE.sub(r"\1", candidate)
            try:
                data = json.loads(normalized)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

            # Fallback: Python literal evaluation, then coerce to JSON-ready types
            try:
                literal = ast.literal_eval(normalized)
            except Exception:
                continue

            jsonable = self._coerce_jsonable(literal)
            if isinstance(jsonable, dict):
                return jsonable

        return None

    def _coerce_jsonable(self, value: Any) -> Any:
        """Convert Python literals into JSON-serializable objects."""
        if isinstance(value, dict):
            return {str(k): self._coerce_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._coerce_jsonable(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)
