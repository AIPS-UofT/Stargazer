from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from .config import NoiseParams, GPParams

def add_white_noise(rv_clean_ms: np.ndarray, sigma_white_ms: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rv_clean_ms + rng.normal(0.0, sigma_white_ms, size=rv_clean_ms.shape)

def add_jitter(sigmas_ms: np.ndarray, sigma_jitter_ms: float) -> np.ndarray:
    return np.sqrt(sigmas_ms**2 + sigma_jitter_ms**2)

def _import_celerite2():
    try:
        import celerite2
        from celerite2 import terms
        return celerite2, terms
    except ImportError as e:
        raise RuntimeError(
            "GP noise requested but 'celerite2' is not installed. Install with: pip install celerite2"
        ) from e

def draw_gp_process(times_days: np.ndarray, gp: GPParams, diag_ms2: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    if not gp.use_gp:
        return np.zeros_like(times_days, dtype=float)
    celerite2, terms = _import_celerite2()
    kernel = terms.RotationTerm(sigma=gp.sigma_ms, period=gp.period_days, Q0=gp.Q0, dQ=gp.dQ, f=gp.f)
    gpmodel = celerite2.GaussianProcess(kernel, mean=0.0)
    times_days = np.asarray(times_days, dtype=float)
    if np.any(np.diff(times_days) < 0):
        raise ValueError("times_days must be sorted ascending for GP computation.")
    gpmodel.compute(times_days, diag=diag_ms2)

    # celerite2's sample() uses numpy's global random state internally
    # If an rng is provided, temporarily seed the global state for reproducibility
    if rng is not None:
        seed = rng.integers(0, 2**31)
        old_state = np.random.get_state()
        np.random.seed(seed)
        sample = gpmodel.sample(size=None, include_mean=True)
        np.random.set_state(old_state)
    else:
        # Use whatever global state was set (e.g., by set_global_seed())
        sample = gpmodel.sample(size=None, include_mean=True)

    return np.array(sample, dtype=float)
