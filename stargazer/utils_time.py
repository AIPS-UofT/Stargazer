from __future__ import annotations
from typing import List
import numpy as np

def sort_times(times_days: List[float]) -> List[float]:
    return list(np.sort(np.array(times_days)))

def baseline_days(times_days: List[float]) -> float:
    t = np.array(times_days)
    if len(t) == 0:
        return 0.0
    return float(t.max() - t.min())
