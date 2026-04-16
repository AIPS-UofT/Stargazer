#!/usr/bin/env python3
"""
Evaluate a single submission against ground truth.
Called by coding agents from their workspace — returns ONLY metrics, never ground truth.

Usage:
    python /path/to/stargazer_submit.py <task_name> <submission_json_path>
    python /path/to/stargazer_submit.py synthetic_seed042 /path/to/submission.json

Output (JSON to stdout):
    {"success": true/false, "reward": 1.23, "metrics": {...}, "hints": [...]}
"""
from __future__ import annotations
import json, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stargazer.bank import TaskBank
from stargazer.env import RvEnv


BANK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stargazer", "stargazer_bank_full")


def evaluate_single(task_name: str, submission_path: str) -> dict:
    bank = TaskBank(BANK_DIR)
    task = bank.load_task(task_name)

    env = RvEnv(task=task, submission_mode="params_and_model", max_steps=3)
    env.reset()

    with open(submission_path) as f:
        submission = json.load(f)

    # Build action
    planets = []
    for p in submission.get("planets", []):
        planets.append({
            "P_days": float(p["P_days"]),
            "m_sin_i_mjup": float(p.get("m_sin_i_mjup", 0.1)),
            "e": float(p.get("e", 0.0)),
            "inc_rad": float(p.get("inc_rad", 0.0)),
            "Omega_rad": float(p.get("Omega_rad", 0.0)),
            "omega_rad": float(p.get("omega_rad", 0.0)),
            "l_rad": float(p.get("l_rad", 0.0)),
        })
    action = {"planets": planets}
    noise = submission.get("noise", {})
    if noise:
        action["noise"] = noise

    _, reward, done, info = env.step(action)

    # Return ONLY metrics — no ground truth parameters
    components = info.get("metrics", {}).get("components", {})
    return {
        "success": bool(info.get("success", False)),
        "reward": round(float(reward), 6),
        "done": bool(done),
        "metrics": {
            "match_score": components.get("match"),
            "delta_bic": components.get("delta_bic"),
            "neg_rms": components.get("neg_rms"),
            "count_penalty": components.get("count"),
        },
        "success_details": info.get("success_details", {}),
        "hints": info.get("hints", []),
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <task_name> <submission.json>", file=sys.stderr)
        sys.exit(1)

    task_name = sys.argv[1]
    submission_path = sys.argv[2]

    try:
        result = evaluate_single(task_name, submission_path)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)
