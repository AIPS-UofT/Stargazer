from __future__ import annotations
from typing import Optional, List
import os, json, glob
from dataclasses import replace

import numpy as np

from .config import Task, PlanetParams, SystemConfig
from .engine_rebound import simulate_clean_rv
from .forward_keplerian import simulate_rv_keplerian
from .task_factory import generate_task


def _apply_rv_only_compat(task: Task) -> Task:
    """Map legacy REBOUND-generated observations to RV-only scoring semantics.

    Replace the REBOUND-generated RV signal with an analytic Keplerian signal,
    keeping the original noise realization. Planet parameters are preserved.
    """
    if not isinstance(task.meta, dict):
        return task
    if task.meta.get("rv_semantics") == "rv_only":
        return task
    if task.meta.get("rv_only_compat_applied"):
        return task
    times = np.asarray(task.observations.times_days, dtype=float)
    if times.size == 0:
        return task

    try:
        rv_clean_old = simulate_clean_rv(task.config, task.observations.times_days)
    except Exception:
        return task

    # Convert planet params to RV-only semantics: l_rad_rv = (l_rad - Omega_rad) % 2pi, Omega_rad = 0
    planets_rv_only = []
    for p in task.config.planets:
        l_rad_rv = (p.l_rad - p.Omega_rad) % (2.0 * np.pi)
        p_rv = replace(p, l_rad=l_rad_rv, Omega_rad=0.0)
        planets_rv_only.append(p_rv)

    inst_gamma = {inst.label: inst.gamma_ms for inst in task.config.instruments}
    gamma_series = np.array(
        [inst_gamma.get(lbl, 0.0) + task.config.star.gamma_ms for lbl in task.observations.instruments],
        dtype=float,
    )
    rv_old_model = rv_clean_old + gamma_series

    rv_new_model = simulate_rv_keplerian(
        planets=planets_rv_only,
        times_days=times,
        M_star_sun=float(task.config.star.M_star_sun),
        gamma_ms=0.0,
    ) + gamma_series

    y_obs = np.asarray(task.observations.rvs_ms, dtype=float)
    residual_noise = y_obs - rv_old_model
    y_new = rv_new_model + residual_noise

    obs_new = replace(task.observations, rvs_ms=y_new.tolist())
    # Update config with RV-only planets
    config_new = replace(task.config, planets=planets_rv_only)
    meta_new = dict(task.meta)
    meta_new["rv_only_compat_applied"] = True
    meta_new["rv_semantics"] = "rv_only_compat"
    return replace(task, config=config_new, observations=obs_new, meta=meta_new)

class TaskBank:
    """File-based bank of tasks for reproducible benchmarks.
    Not thread-safe; intended for single-process use."""
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def add_task(self, task: Task) -> str:
        # naive sanitization: strip path separators
        safe_id = task.task_id.replace(os.sep, "_").replace("/", "_")
        path = os.path.join(self.root_dir, f"{safe_id}.json")
        with open(path, "w") as f:
            f.write(task.to_json())
        return path

    def list_tasks(self) -> List[str]:
        out = []
        for p in glob.glob(os.path.join(self.root_dir, "*.json")):
            try:
                with open(p,"r") as f:
                    Task.from_json(f.read())  # validate
                out.append(os.path.basename(p)[:-5])
            except Exception:
                continue
        return sorted(out)

    def load_task(self, task_id: str) -> Task:
        safe_id = task_id.replace(os.sep, "_").replace("/", "_")
        path = os.path.join(self.root_dir, f"{safe_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Task '{task_id}' not found in bank '{self.root_dir}'")
        with open(path, "r") as f:
            task = Task.from_json(f.read())
        return _apply_rv_only_compat(task)

    def generate_and_add(self, seed: Optional[int] = None, **kwargs) -> Task:
        t = generate_task(seed=seed, **kwargs)
        self.add_task(t)
        return t
