from __future__ import annotations

from typing import List

import numpy as np

from .config import PlanetParams
from .utils_units import semi_amplitude_ms


def _wrap_angle(x: np.ndarray) -> np.ndarray:
    return np.mod(x, 2.0 * np.pi)


def _solve_kepler(M: np.ndarray, ecc: float) -> np.ndarray:
    """Solve E - e sin(E) = M via Newton iterations."""
    e = float(np.clip(ecc, 0.0, 0.999999))
    if e == 0.0:
        return _wrap_angle(M)
    E = _wrap_angle(M + e * np.sin(M))
    for _ in range(80):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        step = f / fp
        E = E - step
        if float(np.max(np.abs(step))) < 1e-12:
            break
    return _wrap_angle(E)


def simulate_rv_keplerian(
    planets: List[PlanetParams],
    times_days: np.ndarray,
    M_star_sun: float,
    gamma_ms: float = 0.0,
) -> np.ndarray:
    """RV-only multi-Keplerian forward model.

    For RV-only tasks (rv_semantics='rv_only'), Omega_rad = 0 and
    l_rad = omega + M0 at t_ref = times_days[0].

    In the general case (REBOUND-generated tasks before rv_only conversion),
    l_rad = Omega + omega + M0.  The code handles both via
    M0 = l_rad - Omega - omega, which reduces to l_rad - omega when
    Omega = 0.  inc_rad is ignored.
    """
    if times_days.size == 0:
        return np.zeros(0, dtype=float)
    t_ref = float(times_days[0])
    rv = np.zeros(times_days.shape, dtype=float)
    for p in planets:
        P = float(p.P_days)
        if P <= 0:
            continue
        e = float(np.clip(p.e, 0.0, 0.95))
        omega = float(p.omega_rad) % (2.0 * np.pi)
        Omega = float(p.Omega_rad) % (2.0 * np.pi)
        # RV-only: l_rad = omega + M0 at t_ref (Omega_rad = 0).
        # General: l_rad = Omega + omega + M0 at t_ref.
        lambda0 = float(p.l_rad) % (2.0 * np.pi)
        M0 = (lambda0 - (Omega + omega)) % (2.0 * np.pi)
        n = 2.0 * np.pi / P
        M = _wrap_angle(M0 + n * (times_days - t_ref))
        E = _solve_kepler(M, e)
        nu = 2.0 * np.arctan2(
            np.sqrt(1.0 + e) * np.sin(E / 2.0),
            np.sqrt(1.0 - e) * np.cos(E / 2.0),
        )
        K = float(semi_amplitude_ms(float(p.m_sin_i_mjup), P, e, float(M_star_sun)))
        rv += K * (np.cos(nu + omega) + e * np.cos(omega))
    return rv + float(gamma_ms)
