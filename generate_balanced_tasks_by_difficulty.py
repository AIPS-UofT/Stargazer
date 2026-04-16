#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

from stargazer.bank import TaskBank
from stargazer.task_factory import generate_task


def _parse_multiplicity_probs(raw: str) -> Tuple[float, float, float, float]:
    parts = [float(x) for x in raw.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("multiplicity_probs must have four comma-separated values")
    total = sum(parts)
    if abs(total - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError(f"multiplicity_probs must sum to 1.0 (got {total})")
    return tuple(parts)  # type: ignore[return-value]


def _parse_range(raw: str) -> Tuple[int, int]:
    parts = [int(x) for x in raw.split(",")]
    if len(parts) != 2 or parts[0] < 1 or parts[0] > parts[1]:
        raise argparse.ArgumentTypeError("n_obs_range must be 'min,max' with min>=1 and min<=max")
    return parts[0], parts[1]


def _parse_difficulties(raw: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            a_s, b_s = token.split("-", 1)
            a, b = int(a_s), int(b_s)
            lo, hi = (a, b) if a <= b else (b, a)
            for d in range(lo, hi + 1):
                if d not in seen:
                    out.append(d)
                    seen.add(d)
        else:
            d = int(token)
            if d not in seen:
                out.append(d)
                seen.add(d)
    if not out:
        raise argparse.ArgumentTypeError("at least one difficulty is required")
    for d in out:
        if d < 1 or d > 10:
            raise argparse.ArgumentTypeError(f"difficulty out of range [1,10]: {d}")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a balanced Stargazer task bank by current truth_difficulty."
    )
    parser.add_argument("--out-dir", default="generated_tasks_balanced_current_difficulty_10x10")
    parser.add_argument("--per-difficulty", type=int, default=10, help="Tasks to keep for each difficulty.")
    parser.add_argument("--difficulties", type=_parse_difficulties, default=list(range(1, 11)))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=200000,
        help="Max seeds to try before giving up.",
    )
    parser.add_argument(
        "--multiplicity-probs",
        type=_parse_multiplicity_probs,
        default=(0.5, 0.3, 0.15, 0.05),
        help="Probabilities for 1,2,3,4 planets (comma-separated).",
    )
    parser.add_argument("--resonance-fraction", type=float, default=0.3)
    parser.add_argument(
        "--n-obs-range",
        type=_parse_range,
        default=(40, 80),
        help="Observation count range 'min,max'.",
    )
    parser.add_argument("--engine", choices=["rebound", "keplerian"], default="rebound")
    parser.add_argument("--los-axis", choices=["x", "y", "z"], default="x")
    parser.add_argument(
        "--integrator-preference",
        choices=["whfast", "ias15"],
        default="whfast",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N attempts.",
    )
    args = parser.parse_args(argv)

    if args.per_difficulty <= 0:
        raise SystemExit("--per-difficulty must be > 0")
    if args.max_attempts <= 0:
        raise SystemExit("--max-attempts must be > 0")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = TaskBank(str(out_dir))

    targets: Dict[int, int] = {d: int(args.per_difficulty) for d in args.difficulties}
    kept: Dict[int, List[dict]] = {d: [] for d in args.difficulties}
    attempts = 0
    seed = args.seed_start

    while attempts < args.max_attempts and any(len(kept[d]) < targets[d] for d in targets):
        task = generate_task(
            seed=seed,
            multiplicity_probs=args.multiplicity_probs,
            resonance_fraction=args.resonance_fraction,
            n_obs_range=args.n_obs_range,
            engine=args.engine,
            los_axis=args.los_axis,
            integrator_preference=args.integrator_preference,
        )
        attempts += 1
        diff = int(task.truth_difficulty)

        if diff in targets and len(kept[diff]) < targets[diff]:
            slot = len(kept[diff]) + 1
            new_task_id = f"seed{seed}_diff{diff}_n{slot:02d}"
            meta = dict(task.meta) if isinstance(task.meta, dict) else {}
            meta["source_seed"] = seed
            meta["source_generated_task_id"] = task.task_id
            task_named = replace(task, task_id=new_task_id, meta=meta)
            bank.add_task(task_named)

            record = {
                "task_id": new_task_id,
                "seed": seed,
                "difficulty": diff,
                "n_planets": len(task_named.config.planets),
                "n_obs": len(task_named.observations.times_days),
                "raw_score": task_named.difficulty_details.get("raw_score"),
            }
            kept[diff].append(record)
            print(
                f"[keep] diff={diff} {len(kept[diff])}/{targets[diff]} "
                f"seed={seed} task_id={new_task_id}"
            )

        if args.progress_every > 0 and attempts % args.progress_every == 0:
            counts = " ".join(f"{d}:{len(kept[d])}/{targets[d]}" for d in sorted(targets))
            print(f"[progress] attempts={attempts} next_seed={seed + 1} {counts}")

        seed += 1

    counts = {str(d): len(kept[d]) for d in sorted(kept)}
    completed = all(len(kept[d]) >= targets[d] for d in targets)
    manifest = {
        "completed": completed,
        "attempts": attempts,
        "seed_start": args.seed_start,
        "next_seed": seed,
        "targets": {str(k): v for k, v in sorted(targets.items())},
        "counts": counts,
        "generation_params": {
            "multiplicity_probs": list(args.multiplicity_probs),
            "resonance_fraction": args.resonance_fraction,
            "n_obs_range": list(args.n_obs_range),
            "engine": args.engine,
            "los_axis": args.los_axis,
            "integrator_preference": args.integrator_preference,
        },
        "tasks_by_difficulty": {str(d): kept[d] for d in sorted(kept)},
    }
    manifest_path = out_dir / "balanced_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Flat text task list for batch scripts.
    task_ids_path = out_dir / "task_ids.txt"
    with open(task_ids_path, "w") as f:
        for d in sorted(kept):
            for item in kept[d]:
                f.write(f"{item['task_id']}\n")

    print(f"\nSaved manifest: {manifest_path}")
    print(f"Saved task IDs: {task_ids_path}")
    print(f"Counts: {counts}")

    if not completed:
        missing = {d: targets[d] - len(kept[d]) for d in sorted(targets) if len(kept[d]) < targets[d]}
        print(f"Failed to reach targets within max_attempts. Missing: {missing}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
