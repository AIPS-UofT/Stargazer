from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np

from .config import PlanetParams, StarParams
from .utils_units import semi_amplitude_ms

# Default weights: rv_curve is dominant (measures actual RV signal agreement),
# dlogP provides alias disambiguation, dlogK/de are secondary diagnostics.
# dphase is zeroed out because it is fully subsumed by rv_curve and subject
# to degeneracies (e.g. omega/M0 trade-off at low eccentricity).
DEFAULT_WEIGHTS: Dict[str, float] = {
    "rv_curve": 4.0,
    "dlogP": 1.0,
    "dlogK": 0.5,
    "de": 0.5,
    "dphase": 0.0,
}

# Legacy weights for when times_days is unavailable (parameter-only matching).
_LEGACY_WEIGHTS: Dict[str, float] = {
    "dlogP": 3.0,
    "dlogK": 2.0,
    "de": 1.0,
    "dphase": 0.5,
}


def _import_linear_sum_assignment():
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment
    except ImportError as e:
        raise RuntimeError("SciPy is required for planet matching: pip install scipy") from e


def planet_score_components(
    truth: PlanetParams,
    guess: PlanetParams,
    M_star_sun: float,
    times_days: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute per-component distances between a truth–guess planet pair.

    When *times_days* is provided the dominant ``rv_curve`` component is
    included: the RMS difference between the single-planet Keplerian RV
    curves, normalised by the truth semi-amplitude.  This naturally
    accounts for parameter degeneracies (e.g. omega/M0 at low *e*).
    """
    if truth.P_days <= 0 or guess.P_days <= 0:
        raise ValueError(f"Period must be positive: truth={truth.P_days}, guess={guess.P_days}")

    K_t = semi_amplitude_ms(truth.m_sin_i_mjup, truth.P_days, truth.e, M_star_sun)
    K_g = semi_amplitude_ms(guess.m_sin_i_mjup, guess.P_days, guess.e, M_star_sun)
    K_t = max(1e-6, K_t)
    K_g = max(1e-6, K_g)

    def wrap_angle(d: float) -> float:
        return (d + np.pi) % (2 * np.pi) - np.pi

    lambda_truth = truth.l_rad - truth.Omega_rad
    lambda_guess = guess.l_rad - guess.Omega_rad

    comp: Dict[str, float] = {
        "dlogP": abs(np.log(guess.P_days / truth.P_days)),
        "dlogK": abs(np.log(K_g / K_t)),
        "de": abs(guess.e - truth.e),
        "dphase": abs(wrap_angle(lambda_guess - lambda_truth)),
    }

    # RV curve distance – the primary matching signal.
    if times_days is not None and times_days.size > 0:
        from .forward_keplerian import simulate_rv_keplerian

        rv_t = simulate_rv_keplerian([truth], times_days, M_star_sun, gamma_ms=0.0)
        rv_g = simulate_rv_keplerian([guess], times_days, M_star_sun, gamma_ms=0.0)
        # Allow an optimal constant offset (gamma) so that only the *shape*
        # of the RV curve matters, not the absolute velocity zero-point.
        offset = float(np.mean(rv_t - rv_g))
        rv_rms = float(np.sqrt(np.mean((rv_t - rv_g - offset) ** 2)))
        comp["rv_curve"] = rv_rms / K_t  # normalised by truth semi-amplitude

    return comp


def planet_distance(
    truth: PlanetParams,
    guess: PlanetParams,
    M_star_sun: float,
    weights: Optional[Dict[str, float]] = None,
    times_days: Optional[np.ndarray] = None,
) -> float:
    comp = planet_score_components(truth, guess, M_star_sun, times_days=times_days)

    if weights is None:
        weights = DEFAULT_WEIGHTS if "rv_curve" in comp else _LEGACY_WEIGHTS

    return float(sum(weights.get(k, 0.0) * comp[k] for k in comp))


def match_planets(
    truth: List[PlanetParams],
    guesses: List[PlanetParams],
    M_star_sun: float,
    weights: Optional[Dict[str, float]] = None,
    max_dist: float = 5.0,
    times_days: Optional[np.ndarray] = None,
) -> Dict[str, any]:
    linear_sum_assignment = _import_linear_sum_assignment()
    n = len(truth)
    m = len(guesses)
    if n == 0 or m == 0:
        return {"pairs": [], "unmatched_truth": list(range(n)), "unmatched_guess": list(range(m))}
    D = np.zeros((n, m), dtype=float)
    for i, t in enumerate(truth):
        for j, g in enumerate(guesses):
            D[i, j] = planet_distance(t, g, M_star_sun, weights=weights, times_days=times_days)
    rows, cols = linear_sum_assignment(D)
    pairs = []
    used_truth = set()
    used_guess = set()
    for i, j in zip(rows, cols):
        if D[i, j] <= max_dist:
            pairs.append((int(i), int(j), float(D[i, j])))
            used_truth.add(int(i))
            used_guess.add(int(j))
    unmatched_truth = [i for i in range(n) if i not in used_truth]
    unmatched_guess = [j for j in range(m) if j not in used_guess]
    return {"pairs": pairs, "unmatched_truth": unmatched_truth, "unmatched_guess": unmatched_guess}
