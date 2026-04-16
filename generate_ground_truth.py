import argparse
import sys
from typing import Tuple
from dataclasses import replace

from stargazer.bank import TaskBank
from stargazer.task_factory import generate_task


def _parse_multiplicity_probs(raw: str) -> Tuple[float, float, float, float]:
    parts = [float(x) for x in raw.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("multiplicity_probs must have four comma-separated values")
    total = sum(parts)
    if abs(total - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError(f"multiplicity_probs must sum to 1.0 (got {total})")
    return tuple(parts)  # type: ignore


def _parse_range(raw: str) -> Tuple[int, int]:
    parts = [int(x) for x in raw.split(",")]
    if len(parts) != 2 or parts[0] < 1 or parts[0] > parts[1]:
        raise argparse.ArgumentTypeError("n_obs_range must be two integers like '40,80' with min<=max and min>=1")
    return parts[0], parts[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Batch-generate Stargazer tasks with ground-truth parameters.")
    parser.add_argument("--out-dir", default="generated_tasks", help="Directory to store generated tasks (JSON files).")
    parser.add_argument("--count", type=int, default=10, help="Number of tasks to generate.")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting seed (increments by 1 for each task).")
    parser.add_argument(
        "--multiplicity-probs",
        type=_parse_multiplicity_probs,
        default=(0.5, 0.3, 0.15, 0.05),
        help="Comma-separated probabilities for 1,2,3,4 planets. Must sum to 1.0.",
    )
    parser.add_argument(
        "--resonance-fraction",
        type=float,
        default=0.3,
        help="Fraction of systems placed near resonance (0-1).",
    )
    parser.add_argument(
        "--n-obs-range",
        type=_parse_range,
        default=(40, 80),
        help="Comma-separated min,max observation counts (e.g., 40,80).",
    )
    parser.add_argument("--engine", choices=["rebound", "keplerian"], default="rebound", help="Physics engine.")
    parser.add_argument("--los-axis", choices=["x", "y", "z"], default="x", help="Line-of-sight axis.")
    parser.add_argument(
        "--integrator-preference",
        choices=["whfast", "ias15"],
        default="whfast",
        help="Rebound integrator preference.",
    )

    args = parser.parse_args(argv)

    bank = TaskBank(args.out_dir)
    for i in range(args.count):
        seed = args.seed_start + i
        task = generate_task(
            seed=seed,
            multiplicity_probs=args.multiplicity_probs,
            resonance_fraction=args.resonance_fraction,
            n_obs_range=args.n_obs_range,
            engine=args.engine,
            los_axis=args.los_axis,
            integrator_preference=args.integrator_preference,
        )
        task_named = replace(task, task_id=f"seed{seed}_diff{task.truth_difficulty}")
        path = bank.add_task(task_named)
        npl = len(task_named.config.planets)
        print(
            f"{i+1:03d}/{args.count} seed={seed} planets={npl} "
            f"difficulty={task_named.truth_difficulty} -> {path}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
