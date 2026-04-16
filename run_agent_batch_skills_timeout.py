#!/usr/bin/env python3
"""Run TabularRvAgent with skills support and hard timeout.

Copy of run_agent_batch_hard_timeout.py with two additions:
  1. Skills documentation is injected into system prompt
  2. Default skills directory is stargazer_skills/

This script is fully isolated — it does NOT modify any existing code.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed

from run_agent_batch import (
    _run_single_task,
    _safe_task_id,
    generate_pass_rate_summary,
    generate_report,
)
from stargazer.bank import TaskBank
from stargazer.limits import DEFAULT_SUBMISSION_MAX_PLANETS

# ---------------------------------------------------------------------------
# Skills support (reused from run_agent_batch_skills.py)
# ---------------------------------------------------------------------------

_DEFAULT_SKILLS_DIR = Path(__file__).parent / "stargazer_skills"
_CORE_SKILLS = [
    "rv_period_search",
    "rv_keplerian_fit",
    "rv_lrad_calculation",
    "rv_submit_strategy",

]
_MULTIPLANET_SKILL = "rv_multiplanet_detection"
_DEFAULT_MULTIPLANET_THRESHOLD = 4


def _load_skill_file(skills_dir: Path, skill_name: str) -> str:
    path = skills_dir / skill_name / "SKILL.md"
    return path.read_text() if path.exists() else ""


def _build_skills_prompt(difficulty: int, skills_dir: Path, multiplanet_threshold: int) -> str:
    names = list(_CORE_SKILLS)
    if difficulty >= multiplanet_threshold:
        names.append(_MULTIPLANET_SKILL)
    sections = [_load_skill_file(skills_dir, n) for n in names]
    sections = [s.strip() for s in sections if s.strip()]
    if not sections:
        return ""
    header = (
        "---\n"
        "## Expert Skills\n\n"
        "The following domain-expert skills are provided to help you succeed. "
        "Read each skill and follow its instructions carefully:\n\n"
    )
    return header + "\n\n---\n\n".join(sections)


def _patch_agent_for_skills(skills_dir: Path, multiplanet_threshold: int) -> None:
    from stargazer.agents.tabular_agent import TabularRvAgent
    _original_build = TabularRvAgent._build_system_prompt

    def _patched_build(self):
        base_prompt = _original_build(self)
        difficulty = 0
        try:
            info = self._current_info
            if isinstance(info, dict):
                difficulty = int(info.get("difficulty", 0))
        except Exception:
            pass
        skills_block = _build_skills_prompt(difficulty, skills_dir, multiplanet_threshold)
        if skills_block:
            return base_prompt + "\n\n" + skills_block
        return base_prompt

    TabularRvAgent._build_system_prompt = _patched_build


def _run_task_subprocess(
    conn,
    task_id: str,
    bank_dir: str,
    model: str,
    batch_dir: Path,
    agent_kwargs: Dict[str, Any],
    skills_dir: Optional[str] = None,
    multiplanet_threshold: int = _DEFAULT_MULTIPLANET_THRESHOLD,
) -> None:
    """Worker entry point for subprocess execution."""
    try:
        if skills_dir:
            _patch_agent_for_skills(Path(skills_dir), multiplanet_threshold)
        result = _run_single_task(task_id, bank_dir, model, batch_dir, agent_kwargs)
    except Exception as exc:  # ensure parent gets an error payload
        result = {
            "task_id": task_id,
            "success": False,
            "reward": 0.0,
            "metrics": {},
            "stop_reason": "worker_exception",
            "error": str(exc),
        }
    conn.send(result)
    conn.close()


def _run_with_watchdog(
    task_id: str,
    bank_dir: str,
    model: str,
    batch_dir: Path,
    agent_kwargs: Dict[str, Any],
    hard_timeout: float,
    skills_dir: Optional[str] = None,
    multiplanet_threshold: int = _DEFAULT_MULTIPLANET_THRESHOLD,
) -> Dict[str, Any]:
    """Execute one task with a subprocess watchdog."""
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(
        target=_run_task_subprocess,
        args=(child_conn, task_id, bank_dir, model, batch_dir, agent_kwargs,
              skills_dir, multiplanet_threshold),
    )
    proc.daemon = True
    proc.start()
    child_conn.close()
    proc.join(hard_timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        parent_conn.close()
        safe_id = _safe_task_id(task_id)
        timeout_dir = batch_dir / f"timeout_{safe_id}"
        timeout_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "task_id": task_id,
            "difficulty": None,
            "n_true_planets": None,
            "n_detected_planets": 0,
            "reward": 0.0,
            "success": False,
            "metrics": {},
            "success_details": {},
            "stop_reason": "hard_timeout",
            "input_tokens_used": 0,
            "output_tokens_used": 0,
            "error": f"Process exceeded {hard_timeout:.1f}s wall-clock limit",
        }
        with open(timeout_dir / "final_result.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(timeout_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write("Task timed out before completion. No trace available.\n")
        return summary

    if parent_conn.poll():
        result = parent_conn.recv()
        parent_conn.close()
        return result

    parent_conn.close()
    return {
        "task_id": task_id,
        "success": False,
        "reward": 0.0,
        "metrics": {},
        "stop_reason": "worker_failed",
        "error": "Worker process exited without returning a result",
    }


def _parse_task_ids(args: argparse.Namespace, bank: TaskBank) -> List[str]:
    if args.task_ids:
        return [x.strip() for x in args.task_ids.split(",") if x.strip()]
    if args.task_ids_file:
        with open(args.task_ids_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if args.start is not None and args.end is not None:
        return bank.list_tasks()[args.start : args.end]
    if args.count:
        return bank.list_tasks()[: args.count]
    print("No task range provided; defaulting to first 3 tasks.")
    return bank.list_tasks()[:3]


def _filter_difficulties(task_ids: List[str], diff_spec: Optional[str], bank: TaskBank) -> List[str]:
    if not diff_spec:
        return task_ids
    allow: set[int] = set()
    for chunk in diff_spec.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            a, b = int(lo), int(hi)
            if a <= b:
                allow.update(range(a, b + 1))
            else:
                allow.update(range(b, a + 1))
        else:
            allow.add(int(token))
    filtered: List[str] = []
    for task_id in task_ids:
        task = bank.load_task(task_id)
        if isinstance(task.truth_difficulty, int) and task.truth_difficulty in allow:
            filtered.append(task_id)
    print(
        f"Difficulty filter active ({sorted(allow)}): {len(filtered)}/{len(task_ids)} tasks kept"
    )
    if not filtered:
        raise ValueError("No tasks left after applying difficulty filter.")
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TabularRvAgent with subprocess timeouts")
    parser.add_argument("--task-ids", type=str, help="Comma-separated task IDs")
    parser.add_argument("--task-ids-file", type=str, help="File with task IDs, one per line")
    parser.add_argument("--count", type=int, help="Run first N task IDs from bank")
    parser.add_argument("--start", type=int, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, help="End index (exclusive)")
    parser.add_argument("--bank-dir", default="stargazer/Stargazer_synthetic_task", help="TaskBank directory")
    parser.add_argument("--output-dir", default="batch_results_hard_timeout", help="Output directory")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--resume", action="store_true", help="Skip tasks that already have results in output-dir")
    parser.add_argument("--model", default="gpt-5-mini", help="Model name")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tool-calls", type=int, default=20)
    parser.add_argument("--max-execution-time", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=300000)
    parser.add_argument("--max-planets", type=int, default=DEFAULT_SUBMISSION_MAX_PLANETS)
    parser.add_argument("--submission-mode", type=str, default="params_and_model")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--api-timeout", type=float, default=60.0)
    parser.add_argument(
        "--difficulty-budget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-adjust max tokens/steps per difficulty tier",
    )
    parser.add_argument("--reasoning-effort", type=str, choices=["low", "medium", "high"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--hard-timeout",
        type=float,
        default=1600.0,
        help="Wall-clock seconds before forcibly terminating a task subprocess.",
    )
    parser.add_argument(
        "--difficulties",
        type=str,
        default=None,
        help="Filter tasks by difficulty, e.g. '2' or '1-5,8'.",
    )
    parser.add_argument(
        "--skills",
        action="store_true",
        default=True,
        help="Inject skills into the agent system prompt (default: True).",
    )
    parser.add_argument(
        "--skills-dir",
        type=str,
        default=str(_DEFAULT_SKILLS_DIR),
        help="Path to the skills directory (default: stargazer_skills/).",
    )
    parser.add_argument(
        "--multiplanet-threshold",
        type=int,
        default=_DEFAULT_MULTIPLANET_THRESHOLD,
        help="Difficulty >= this adds the multi-planet skill.",
    )

    args = parser.parse_args()

    bank = TaskBank(args.bank_dir)
    task_ids = _parse_task_ids(args, bank)
    task_ids = _filter_difficulties(task_ids, args.difficulties, bank)

    batch_dir = Path(args.output_dir)

    # --resume: skip tasks that already have a final_result.json
    if args.resume:
        before = len(task_ids)
        completed_task_ids: set[str] = set()
        if batch_dir.exists():
            for p in batch_dir.iterdir():
                if not p.is_dir():
                    continue
                fr = p / "final_result.json"
                if fr.exists():
                    try:
                        with open(fr) as _f:
                            tid = json.load(_f).get("task_id")
                            if tid:
                                completed_task_ids.add(tid)
                    except Exception:
                        pass
        task_ids = [tid for tid in task_ids if tid not in completed_task_ids]
        skipped = before - len(task_ids)
        print(f"[Resume] Skipping {skipped} already-completed tasks, {len(task_ids)} remaining")
    batch_dir.mkdir(parents=True, exist_ok=True)

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
    }

    batch_config = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "bank_dir": args.bank_dir,
        "task_ids": task_ids,
        "n_tasks": len(task_ids),
        "hard_timeout": args.hard_timeout,
        "agent_kwargs": agent_kwargs,
    }
    with open(batch_dir / "batch_config.json", "w", encoding="utf-8") as f:
        json.dump(batch_config, f, indent=2)

    results_map: Dict[str, Dict[str, Any]] = {}

    skills_dir = args.skills_dir if args.skills else None
    if args.skills:
        sd = Path(args.skills_dir)
        loaded = [n for n in _CORE_SKILLS + [_MULTIPLANET_SKILL]
                  if (sd / n / "SKILL.md").exists()]
        print(f"[Skills] Loaded {len(loaded)} skill(s) from {sd}: {loaded}")
        print(f"[Skills] Multi-planet skill active for difficulty >= {args.multiplanet_threshold}")

    def _run_single(task_id: str) -> Dict[str, Any]:
        return _run_with_watchdog(
            task_id,
            args.bank_dir,
            args.model,
            batch_dir,
            agent_kwargs,
            args.hard_timeout,
            skills_dir=skills_dir,
            multiplanet_threshold=args.multiplanet_threshold,
        )

    if args.workers <= 1:
        for idx, task_id in enumerate(task_ids, 1):
            print(f"\n[{idx}/{len(task_ids)}] Running task_id={task_id}")
            result = _run_single(task_id)
            results_map[task_id] = result
            print(
                f"  -> success={result.get('success', False)}, "
                f"stop_reason={result.get('stop_reason')}, "
                f"reward={float(result.get('reward', 0.0)):.4f}"
            )
    else:
        print(f"[Info] Launching up to {args.workers} parallel worker(s) with hard timeouts")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run_single, task_id): task_id for task_id in task_ids}
            done = 0
            total = len(futures)
            for future in as_completed(futures):
                task_id = futures[future]
                done += 1
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "task_id": task_id,
                        "success": False,
                        "reward": 0.0,
                        "metrics": {},
                        "stop_reason": "worker_exception",
                        "error": str(exc),
                    }
                results_map[task_id] = result
                print(
                    f"[{done}/{total}] {task_id}: success={result.get('success', False)}, "
                    f"stop_reason={result.get('stop_reason')}, "
                    f"reward={float(result.get('reward', 0.0)):.4f}"
                )

    results: List[Dict[str, Any]] = [results_map[task_id] for task_id in task_ids]

    results_file = batch_dir / "batch_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    generate_report(results, batch_dir)
    generate_pass_rate_summary(results, batch_dir)

    passed = sum(1 for r in results if r.get("success", False))
    print("=" * 70)
    print("Batch complete (hard timeout mode)")
    print(f"Passed: {passed}/{len(results)}")
    print(f"Output directory: {batch_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        sys.exit(1)
