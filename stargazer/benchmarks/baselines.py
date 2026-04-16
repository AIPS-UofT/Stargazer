from __future__ import annotations
from typing import Dict, Any
import numpy as np

REQUIRED_OBS_KEYS = ("times_days", "rvs_ms", "sigmas_ms")


def _validate_observation(observation: Dict[str, Any], fn_name: str) -> None:
    if not isinstance(observation, dict):
        raise TypeError(
            f"{fn_name} expects a dict observation with keys {REQUIRED_OBS_KEYS}, "
            f"got {type(observation).__name__}"
        )
    missing = [k for k in REQUIRED_OBS_KEYS if k not in observation]
    if missing:
        raise KeyError(f"{fn_name} missing required keys: {missing}")


def baseline_null_model(observation: Dict[str, Any]) -> Dict[str, Any]:
    _validate_observation(observation, "baseline_null_model")
    t = np.array(observation["times_days"], dtype=float)
    y = np.array(observation["rvs_ms"], dtype=float)
    s_arr = np.array(observation["sigmas_ms"], dtype=float)
    if y.size == 0:
        raise ValueError("Empty observations")
    if np.any(~np.isfinite(s_arr)) or np.any(s_arr <= 0):
        raise ValueError("All per-point uncertainties must be positive and finite")
    w = 1.0 / (s_arr**2)
    mu = float(np.sum(w*y)/np.sum(w))
    rv_model = np.full_like(y, mu)
    return {"rv_model": rv_model.tolist(), "planets": []}

def baseline_one_sine(observation: Dict[str, Any]) -> Dict[str, Any]:
    _validate_observation(observation, "baseline_one_sine")
    t = np.array(observation["times_days"], dtype=float)
    y = np.array(observation["rvs_ms"], dtype=float)
    s = np.array(observation["sigmas_ms"], dtype=float)
    if y.size == 0:
        raise ValueError("Empty observations")
    if np.any(~np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("All per-point uncertainties must be positive and finite")

    weights = 1.0 / (s**2 + 1e-6)
    y_mean = float(np.average(y, weights=weights))
    y0 = y - y_mean
    freqs = np.linspace(1/300.0, 1/2.0, 2000)
    best = None
    for f in freqs:
        omega = 2*np.pi*f
        X = np.vstack([np.sin(omega*t), np.cos(omega*t), np.ones_like(t)]).T
        WX = X * weights[:, None]
        beta = np.linalg.pinv(X.T @ WX) @ (X.T @ (weights * y0))
        model = X @ beta + y_mean
        rss = np.sum(((y - model)/s)**2)
        if best is None or rss < best[0]:
            best = (rss, f, model, beta)
    rv_model = best[2]
    P_days = 1.0/best[1]
    beta_best = best[3]
    amp = float(np.hypot(beta_best[0], beta_best[1]))
    omega = 2.0 * np.pi * best[1]
    # Model is y = gamma + A*sin(w t) + B*cos(w t).
    # Convert to y = gamma + K*cos(M), where M = M0 + w*(t-t_ref), e=0.
    # Using sin(wt+phi) form, M0 = phi - pi/2 + w*t_ref.
    phase_sine = float(np.arctan2(beta_best[1], beta_best[0]))
    t_ref = float(t[0])
    M0 = float((phase_sine - np.pi / 2.0 + omega * t_ref) % (2.0 * np.pi))
    rv_offset_ms = float(y_mean + beta_best[2])
    resid = y - rv_model
    rms_ms = float(np.sqrt(np.mean(resid**2)))
    wrms_ms = float(np.sqrt(np.average(resid**2, weights=weights)))
    P_years = P_days/365.25
    m_sin_i_estimate = amp / (28.4329 * (P_years ** (-1.0/3.0)))
    m_sin_i_estimate = float(np.clip(m_sin_i_estimate, 0.001, 10.0))
    guess = {"P_days": float(P_days), "m_sin_i_mjup": m_sin_i_estimate, "e": 0.0,
             "inc_rad": float(np.pi/2), "Omega_rad": 0.0, "omega_rad": 0.0, "l_rad": M0}
    return {
        "rv_model": rv_model.tolist(),
        "planets": [guess],
        "period_days": float(P_days),
        "semi_amplitude_ms": amp,
        "phase_rad": M0,
        "rv_offset_ms": rv_offset_ms,
        "rms_ms": rms_ms,
        "wrms_ms": wrms_ms,
    }
