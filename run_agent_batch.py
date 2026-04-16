#!/usr/bin/env python3
"""
Batch runner for TabularRvAgent on Stargazer TaskBank.

Examples:
    python run_agent_batch.py --count 10 --bank-dir generated_tasks_balanced_75
    python run_agent_batch.py --task-ids seed1_diff8,seed20_diff3 --workers 4
    python run_agent_batch.py --task-ids-file task_ids.txt --max-execution-time 900
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from stargazer.bank import TaskBank
from stargazer.env import RvEnv
from stargazer.agents import TabularRvAgent, TabularAgentConfig
from stargazer.agents.format_utils import convert_trace_to_json, trace_to_markdown, trace_to_html
from stargazer.limits import DEFAULT_SUBMISSION_MAX_PLANETS


def _safe_task_id(task_id: str) -> str:
    return task_id.replace("/", "_")


def _parse_difficulties(spec: str) -> set[int]:
    """
    Parse difficulty spec like:
      "2"
      "1,2,3"
      "1-5,8,10"
    """
    out: set[int] = set()
    for chunk in spec.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            a_str, b_str = token.split("-", 1)
            a = int(a_str.strip())
            b = int(b_str.strip())
            lo, hi = (a, b) if a <= b else (b, a)
            out.update(range(lo, hi + 1))
        else:
            out.add(int(token))
    return out


def _write_task_files(
    task_dir: Path,
    task_id: str,
    model: str,
    json_trace: Dict[str, Any],
    final_result: Dict[str, Any],
) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"trace_{_safe_task_id(task_id)}_{model.replace('/', '_')}"
    json_path = task_dir / f"{base_name}.json"
    md_path = task_dir / f"{base_name}.md"
    html_path = task_dir / f"{base_name}.html"
    final_path = task_dir / "final_result.json"
    summary_path = task_dir / "summary.txt"

    with open(json_path, "w") as f:
        json.dump(json_trace, f, indent=2)
    with open(md_path, "w") as f:
        f.write(trace_to_markdown(json_trace))
    with open(html_path, "w") as f:
        f.write(trace_to_html(json_trace))
    with open(final_path, "w") as f:
        json.dump(final_result, f, indent=2)

    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Stargazer Tabular Batch Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Task ID: {task_id}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Success: {final_result.get('success', False)}\n")
        f.write(f"Reward: {final_result.get('reward', 0):.6f}\n")
        f.write(f"Stop reason: {final_result.get('stop_reason')}\n")
        f.write(
            f"Tokens: in={final_result.get('input_tokens_used', 0)}, "
            f"out={final_result.get('output_tokens_used', 0)}\n"
        )
        if final_result.get("error"):
            f.write(f"Error: {final_result['error']}\n")


def _run_single_task(
    task_id: str,
    bank_dir: str,
    model: str,
    batch_dir: Path,
    agent_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        bank = TaskBank(bank_dir)
        task = bank.load_task(task_id)

        tier = _difficulty_tier(task.truth_difficulty)
        max_tokens = agent_kwargs["max_tokens"]
        max_steps = agent_kwargs["max_steps"]
        max_time = agent_kwargs["max_execution_time"]
        if agent_kwargs.get("use_difficulty_budget", False) and tier:
            budget = _DIFFICULTY_BUDGETS[tier]
            max_tokens = budget["max_tokens"]
            max_steps = budget["max_steps"]
            max_time = budget["max_time"]
            if agent_kwargs.get("verbose"):
                print(
                    f"[Budget] Difficulty {task.truth_difficulty} ({tier}) "
                    f"→ tokens={max_tokens}, steps={max_steps}, time={max_time}s"
                )

        env = RvEnv(
            task=task,
            submission_mode=agent_kwargs["submission_mode"],
            max_steps=max_steps,
        )

        config = TabularAgentConfig(
            model=model,
            temperature=agent_kwargs["temperature"],
            max_tool_calls=agent_kwargs["max_tool_calls"],
            max_execution_time=max_time,
            max_tokens_per_task=max_tokens,
            max_planets=agent_kwargs["max_planets"],
            reasoning_effort=agent_kwargs["reasoning_effort"],
            stream_output=agent_kwargs["stream"],
            api_timeout_sec=agent_kwargs["api_timeout"],
        )

        # Create task output directory early so incremental traces can be written
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        task_dir = batch_dir / f"{ts}_{_safe_task_id(task.task_id)}"
        task_dir.mkdir(parents=True, exist_ok=True)

        # Create agent — with optional mentor
        if agent_kwargs.get("mentor"):
            from stargazer.agents.mentor import MentorConfig
            from stargazer.agents.mentored_runner import create_mentored_agent

            mentor_config = MentorConfig(
                model=agent_kwargs.get("mentor_model", model),
                intervention_policy=agent_kwargs.get("mentor_policy", "key_decisions"),
                max_mentor_calls=agent_kwargs.get("mentor_max_calls", 5),
                max_mentor_tokens=agent_kwargs.get("mentor_max_tokens", 50000),
            )
            agent = create_mentored_agent(env, worker_config=config, mentor_config=mentor_config, trace_dir=task_dir)
        else:
            agent = TabularRvAgent(env, config=config, trace_dir=task_dir)

        trace = agent.run(verbose=agent_kwargs["verbose"])
        json_trace = convert_trace_to_json(trace)

        history = json_trace.get("history", [])
        last = history[-1] if history else {}
        metrics = last.get("metrics", {}) if isinstance(last, dict) else {}
        components = metrics.get("components", {}) if isinstance(metrics, dict) else {}
        submission = metrics.get("submission", {}) if isinstance(metrics, dict) else {}
        planets = submission.get("planets", []) if isinstance(submission, dict) else []

        result = {
            "task_id": task.task_id,
            "difficulty": task.truth_difficulty,
            "n_true_planets": len(task.config.planets),
            "n_detected_planets": len(planets) if isinstance(planets, list) else 0,
            "reward": float(last.get("reward", 0.0)) if isinstance(last, dict) else 0.0,
            "success": bool(last.get("success", False)) if isinstance(last, dict) else False,
            "metrics": components,
            "success_details": last.get("success_details", {}) if isinstance(last, dict) else {},
            "stop_reason": json_trace.get("stop_reason"),
            "input_tokens_used": json_trace.get("input_tokens_used", 0),
            "output_tokens_used": json_trace.get("output_tokens_used", 0),
            "error": json_trace.get("error_message"),
        }

        _write_task_files(task_dir, task.task_id, model, json_trace, result)
        result["output_dir"] = str(task_dir)
        return result

    except Exception as e:
        return {
            "task_id": task_id,
            "success": False,
            "error": str(e),
            "reward": 0.0,
            "metrics": {},
        }


def generate_pass_rate_summary(results: List[Dict[str, Any]], output_dir: Path) -> None:
    """Generate machine-readable pass-rate summary."""
    total = len(results)
    passed = sum(1 for r in results if r.get("success", False))
    failed = total - passed
    errored = sum(1 for r in results if r.get("error"))

    by_difficulty = defaultdict(list)
    unknown_difficulty = []
    for r in results:
        diff = r.get("difficulty")
        if isinstance(diff, int):
            by_difficulty[diff].append(r)
        else:
            unknown_difficulty.append(r)

    by_diff_summary = {}
    for diff, rows in sorted(by_difficulty.items()):
        n = len(rows)
        n_pass = sum(1 for x in rows if x.get("success", False))
        n_fail = n - n_pass
        by_diff_summary[str(diff)] = {
            "total": n,
            "passed": n_pass,
            "failed": n_fail,
            "pass_rate": (n_pass / n) if n else 0.0,
        }

    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_tasks": total,
        "passed": passed,
        "failed": failed,
        "errored": errored,
        "overall_pass_rate": (passed / total) if total else 0.0,
        "by_difficulty": by_diff_summary,
        "unknown_difficulty_tasks": len(unknown_difficulty),
    }

    out_file = output_dir / "pass_rate_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved pass-rate summary: {out_file}")


def generate_report(results: List[Dict[str, Any]], output_dir: Path) -> None:
    report_file = output_dir / "batch_report.txt"
    with open(report_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Batch Run Report\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total tasks: {len(results)}\n\n")

        successful = [r for r in results if r.get("success", False)]
        success_rate = (100 * len(successful) / len(results)) if results else 0.0
        f.write(f"Success rate: {len(successful)}/{len(results)} ({success_rate:.1f}%)\n\n")

        f.write("By difficulty:\n")
        by_difficulty = defaultdict(list)
        for r in results:
            if isinstance(r.get("difficulty"), int):
                by_difficulty[r["difficulty"]].append(r)
        for diff in sorted(by_difficulty):
            tasks = by_difficulty[diff]
            s_rate = sum(1 for t in tasks if t.get("success", False)) / len(tasks)
            avg_reward = sum(float(t.get("reward", 0.0)) for t in tasks) / len(tasks)
            f.write(
                f"  Diff {diff}: {len(tasks)} tasks, "
                f"success {s_rate:.1%}, avg reward {avg_reward:.4f}\n"
            )

        f.write("\nDetails:\n")
        f.write("-" * 70 + "\n")
        for i, r in enumerate(results, 1):
            f.write(f"\nTask {i}:\n")
            f.write(f"  task_id: {r.get('task_id', 'N/A')}\n")
            f.write(f"  difficulty: {r.get('difficulty', 'N/A')}\n")
            f.write(f"  success: {r.get('success', False)}\n")
            f.write(f"  reward: {float(r.get('reward', 0.0)):.4f}\n")
            if r.get("error"):
                f.write(f"  error: {r['error']}\n")
    print(f"Saved report: {report_file}")


# Input token price per million tokens by model family (USD)
_INPUT_PRICE_PER_MTOK: Dict[str, float] = {
    "claude": 3.0,      # claude-sonnet-4.5
    "gpt-5-mini": 0.40,
    "gpt-5": 2.0,
    "gpt-4o-mini": 0.15,
    "gpt-4o": 2.50,
    "o1": 15.0,
    "o3": 10.0,
    "intern": 0.0,      # InternLM (chat.intern-ai.org.cn) - free API
    "openrouter": 1.0,  # OpenRouter - conservative estimate; actual price varies by routed model
}

_DIFFICULTY_BUDGETS = {
    "easy": {"max_tokens": 200_000, "max_steps": 3, "max_time": 600},
    "medium": {"max_tokens": 450_000, "max_steps": 5, "max_time": 900},
    "hard": {"max_tokens": 900_000, "max_steps": 10, "max_time": 1500},
}


def _difficulty_tier(difficulty: Optional[int]) -> Optional[str]:
    if not isinstance(difficulty, int):
        return None
    if difficulty <= 2:
        return "easy"
    if 3 <= difficulty <= 6:
        return "medium"
    if difficulty >= 7:
        return "hard"
    return None

def _input_price_per_mtok(model: str) -> float:
    """Return the input price per million tokens for the given model."""
    m = model.lower()
    for key, price in _INPUT_PRICE_PER_MTOK.items():
        if m.startswith(key):
            return price
    return 3.0  # conservative default


def run_batch(
    task_ids: List[str],
    output_dir: str,
    model: str,
    bank_dir: str,
    workers: int,
    agent_kwargs: Dict[str, Any],
    max_total_tokens: int = 0,
    max_cost_usd: float = 0.0,
) -> None:
    # Convert cost cap to token cap (based on input price, conservative)
    if max_cost_usd > 0 and max_total_tokens <= 0:
        price = _input_price_per_mtok(model)
        max_total_tokens = int(max_cost_usd / price * 1_000_000)
        print(f"[Budget] --max-cost-usd={max_cost_usd} → token cap={max_total_tokens:,} "
              f"(at ${price}/MTok input for {model})")
    bank = TaskBank(bank_dir)
    available_task_ids = set(bank.list_tasks())
    missing = [task_id for task_id in task_ids if task_id not in available_task_ids]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} task_id(s) not found in bank '{bank_dir}': {missing}"
        )

    batch_dir = Path(output_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)

    batch_config = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "bank_dir": bank_dir,
        "task_ids": task_ids,
        "n_tasks": len(task_ids),
        "workers": workers,
        "max_total_tokens": max_total_tokens,
        "agent_kwargs": agent_kwargs,
    }
    with open(batch_dir / "batch_config.json", "w") as f:
        json.dump(batch_config, f, indent=2)

    print("=" * 70)
    print(f"Running TabularRvAgent batch: {len(task_ids)} task(s)")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Bank: {bank_dir}")
    print(f"Workers: {workers}")
    print(f"Output: {batch_dir}")
    if max_total_tokens > 0:
        print(f"Global token cap: {max_total_tokens:,}")
    print("=" * 70)

    results_map: Dict[str, Dict[str, Any]] = {}

    # Global token counter (thread-safe)
    _token_lock = threading.Lock()
    _total_tokens_used = [0]  # mutable container for nonlocal mutation in nested fn

    def _budget_ok() -> bool:
        if max_total_tokens <= 0:
            return True
        with _token_lock:
            return _total_tokens_used[0] < max_total_tokens

    def _record_tokens(result: Dict[str, Any]) -> None:
        used = result.get("input_tokens_used", 0) + result.get("output_tokens_used", 0)
        with _token_lock:
            _total_tokens_used[0] += used

    if workers <= 1:
        for i, task_id in enumerate(task_ids, 1):
            if not _budget_ok():
                print(
                    f"\n[Budget] Global token cap {max_total_tokens:,} reached "
                    f"({_total_tokens_used[0]:,} used). Stopping after {i-1} tasks."
                )
                break
            print(f"\n[{i}/{len(task_ids)}] Running task_id={task_id}")
            result = _run_single_task(task_id, bank_dir, model, batch_dir, agent_kwargs)
            _record_tokens(result)
            results_map[task_id] = result
            print(
                f"  -> success={result.get('success', False)}, "
                f"reward={float(result.get('reward', 0.0)):.4f}, "
                f"tokens_total={_total_tokens_used[0]:,}"
                + (f"/{max_total_tokens:,}" if max_total_tokens > 0 else "")
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {}
            for task_id in task_ids:
                if not _budget_ok():
                    print(
                        f"[Budget] Global token cap {max_total_tokens:,} reached "
                        f"({_total_tokens_used[0]:,} used). Not submitting more tasks."
                    )
                    break
                futures[ex.submit(
                    _run_single_task,
                    task_id,
                    bank_dir,
                    model,
                    batch_dir,
                    agent_kwargs,
                )] = task_id
            done_count = 0
            for future in as_completed(futures):
                task_id = futures[future]
                done_count += 1
                try:
                    result = future.result()
                except Exception as e:
                    result = {"task_id": task_id, "success": False, "error": str(e), "reward": 0.0}
                _record_tokens(result)
                results_map[task_id] = result
                print(
                    f"[{done_count}/{len(futures)}] {task_id}: "
                    f"success={result.get('success', False)}, "
                    f"reward={float(result.get('reward', 0.0)):.4f}, "
                    f"tokens_total={_total_tokens_used[0]:,}"
                    + (f"/{max_total_tokens:,}" if max_total_tokens > 0 else "")
                )

    # Only include tasks that were actually run (budget may have cut short)
    results = [results_map[task_id] for task_id in task_ids if task_id in results_map]
    skipped = len(task_ids) - len(results)

    results_file = batch_dir / "batch_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    generate_report(results, batch_dir)
    generate_pass_rate_summary(results, batch_dir)

    passed = sum(1 for r in results if r.get("success", False))
    print("=" * 70)
    print("Batch complete")
    print(f"Passed: {passed}/{len(results)} ({(passed/len(results)*100) if results else 0.0:.1f}%)")
    if skipped:
        print(f"Skipped: {skipped} tasks (budget exceeded)")
    print(f"Output directory: {batch_dir.resolve()}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TabularRvAgent in batch mode")
    parser.add_argument("--task-ids", type=str, help="Comma-separated task IDs")
    parser.add_argument("--task-ids-file", type=str, help="File with task IDs, one per line")
    parser.add_argument("--count", type=int, help="Run first N task IDs from bank")
    parser.add_argument("--start", type=int, help="Start index (inclusive) in sorted task IDs")
    parser.add_argument("--end", type=int, help="End index (exclusive) in sorted task IDs")

    parser.add_argument("--output-dir", default="batch_results", help="Output directory")
    parser.add_argument("--bank-dir", default="stargazer/Stargazer_synthetic_task", help="TaskBank directory")
    parser.add_argument("--model", default="gpt-5-mini", help="Model name")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument(
        "--difficulties",
        type=str,
        default=None,
        help="Filter by difficulty, e.g. '2' or '1-5,8,10'",
    )

    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--max-tool-calls", type=int, default=20, help="Max tool calls")
    parser.add_argument("--max-execution-time", type=float, default=300.0, help="Max seconds per task")
    parser.add_argument("--max-tokens", type=int, default=1000000, help="Max tokens per task")
    parser.add_argument(
        "--max-planets",
        type=int,
        default=DEFAULT_SUBMISSION_MAX_PLANETS,
        help="Max planets in submission. Default matches the 7-planet real-task upper bound.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="Reasoning effort for O-series models",
    )
    parser.add_argument(
        "--submission-mode",
        type=str,
        default="params_and_model",
        choices=["model_only", "params_only", "params_and_model"],
        help="Submission mode",
    )
    parser.add_argument("--max-steps", type=int, default=3, help="Max env submission attempts per task")
    parser.add_argument("--api-timeout", type=float, default=60.0, help="API timeout in seconds")
    parser.add_argument(
        "--difficulty-budget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-adjust max tokens/steps per difficulty tier (easy/medium/hard)",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=0,
        help="Global token budget for the entire batch (0 = unlimited). "
             "Stops submitting new tasks once exceeded.",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=0.0,
        help="Global cost cap in USD (0 = unlimited). Converted to a token cap "
             "based on the model's input price. Claude-aware.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose per-task logs")
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stream assistant text output",
    )

    # Mentor agent options
    parser.add_argument(
        "--mentor",
        action="store_true",
        default=False,
        help="Enable mentor agent to guide the worker",
    )
    parser.add_argument(
        "--mentor-model",
        type=str,
        default=None,
        help="Model for mentor agent (defaults to same as --model)",
    )
    parser.add_argument(
        "--mentor-policy",
        type=str,
        default="key_decisions",
        choices=["every_step", "key_decisions", "pre_submit_only"],
        help="Mentor intervention policy",
    )
    parser.add_argument(
        "--mentor-max-calls",
        type=int,
        default=5,
        help="Max mentor interventions per task",
    )
    parser.add_argument(
        "--mentor-max-tokens",
        type=int,
        default=50000,
        help="Max total tokens for mentor per task",
    )

    args = parser.parse_args()

    bank = TaskBank(args.bank_dir)
    all_task_ids = bank.list_tasks()

    if args.task_ids:
        task_ids = [x.strip() for x in args.task_ids.split(",") if x.strip()]
    elif args.task_ids_file:
        with open(args.task_ids_file, "r") as f:
            task_ids = [line.strip() for line in f if line.strip()]
    elif args.start is not None and args.end is not None:
        task_ids = all_task_ids[args.start : args.end]
    elif args.count:
        task_ids = all_task_ids[: args.count]
    else:
        task_ids = all_task_ids[:3]
        print("No task range provided; defaulting to first 3 tasks.")

    # Optional difficulty filter over selected task_ids
    if args.difficulties:
        allow = _parse_difficulties(args.difficulties)
        filtered: List[str] = []
        for task_id in task_ids:
            task = bank.load_task(task_id)
            if isinstance(task.truth_difficulty, int) and task.truth_difficulty in allow:
                filtered.append(task_id)
        print(
            f"Difficulty filter active ({sorted(allow)}): "
            f"{len(filtered)}/{len(task_ids)} tasks kept"
        )
        task_ids = filtered
        if not task_ids:
            raise ValueError("No tasks left after applying --difficulties filter.")

    agent_kwargs = {
        "temperature": args.temperature,
        "max_tool_calls": args.max_tool_calls,
        "max_execution_time": args.max_execution_time,
        "max_tokens": args.max_tokens,
        "max_planets": args.max_planets,
        "reasoning_effort": args.reasoning_effort,
        "submission_mode": args.submission_mode,
        "max_steps": args.max_steps,
        "api_timeout": args.api_timeout,
        "verbose": args.verbose,
        "stream": args.stream,
        "use_difficulty_budget": args.difficulty_budget,
        "mentor": args.mentor,
        "mentor_model": args.mentor_model or args.model,
        "mentor_policy": args.mentor_policy,
        "mentor_max_calls": args.mentor_max_calls,
        "mentor_max_tokens": args.mentor_max_tokens,
    }

    run_batch(
        task_ids=task_ids,
        output_dir=args.output_dir,
        model=args.model,
        bank_dir=args.bank_dir,
        workers=max(1, args.workers),
        agent_kwargs=agent_kwargs,
        max_total_tokens=args.max_total_tokens,
        max_cost_usd=args.max_cost_usd,
    )


if __name__ == "__main__":
    main()
