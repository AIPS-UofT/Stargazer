#!/usr/bin/env python3
"""
One-time script: generate 100 synthetic tasks and save to a TaskBank.
This runs REBOUND simulations (slow, ~5-10 min), but only needs to run ONCE.

Usage:
    python generate_synthetic_bank.py [--bank-dir stargazer/stargazer_bank] [--n 100]
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from stargazer.task_factory import TaskFactory
from stargazer.bank import TaskBank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-dir", default="stargazer/Stargazer_synthetic_task")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    bank = TaskBank(args.bank_dir)
    factory = TaskFactory()

    existing = set(bank.list_tasks())
    for seed in range(args.n):
        tag = f"synthetic_seed{seed:03d}"
        if tag in existing:
            print(f"  [SKIP] {tag} already exists")
            continue
        print(f"  Generating {tag} ...", end=" ", flush=True)
        task = factory.sample(seed=seed)
        # Override task_id to a human-readable name
        from dataclasses import replace
        task = replace(task, task_id=tag)
        bank.add_task(task)
        print(f"done (diff={task.truth_difficulty}, {len(task.observations.times_days)} obs)")

    print(f"\nBank now has {len(bank.list_tasks())} tasks in {args.bank_dir}")


if __name__ == "__main__":
    main()
