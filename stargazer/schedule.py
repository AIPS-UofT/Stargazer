from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np

from .config import ObservingSchedule

def random_uniform_over_baseline(inner_period_days: float,
                                 n_obs: int = 50,
                                 baseline_multiplier_range: Tuple[float,float] = (2.0, 4.0),
                                 instrument_label: str = "instA",
                                 seed: Optional[int] = None) -> ObservingSchedule:
    if inner_period_days <= 0:
        raise ValueError(f"inner_period_days must be >0, got {inner_period_days}")
    if not isinstance(n_obs, int) or n_obs <= 0:
        raise ValueError(f"n_obs must be positive int, got {n_obs}")
    if not (isinstance(baseline_multiplier_range, tuple) and len(baseline_multiplier_range)==2):
        raise ValueError("baseline_multiplier_range must be a (lo, hi) tuple")
    lo, hi = baseline_multiplier_range
    if lo <= 0 or hi <= 0 or lo > hi:
        raise ValueError(f"Invalid baseline_multiplier_range: {baseline_multiplier_range}")
    rng = np.random.default_rng(seed)
    baseline_days = float(inner_period_days * rng.uniform(*baseline_multiplier_range))
    times = np.sort(rng.uniform(0.0, baseline_days, size=n_obs))
    instruments = [instrument_label] * n_obs
    return ObservingSchedule(times_days=times.tolist(), instruments=instruments)
