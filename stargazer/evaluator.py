from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
import numpy as np, math

from .config import SystemConfig, Observations, PlanetParams
from .forward_keplerian import simulate_rv_keplerian
from .matching import match_planets

def loglike_white_jitter(rv_obs: np.ndarray, rv_model: np.ndarray, sigma_obs: np.ndarray, sigma_jitter: float = 0.0) -> float:
    var = sigma_obs**2 + sigma_jitter**2 + 1e-12
    if np.any(var <= 0) or np.any(~np.isfinite(var)):
        raise ValueError("Non-positive or non-finite variance encountered in likelihood.")
    resid2 = (rv_obs - rv_model)**2
    ll = -0.5 * np.sum(resid2/var + np.log(2*np.pi*var))
    return float(ll)

def best_constant_fit(
    rv_obs: np.ndarray,
    sigma_obs: np.ndarray,
    instruments: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Fit a constant (per-instrument) mean as the null model."""
    model = np.empty_like(rv_obs)
    if instruments is not None:
        unique_inst = np.unique(instruments)
        k = len(unique_inst)
        mu_per_inst: Dict[str, float] = {}
        for inst in unique_inst:
            mask = instruments == inst
            w = 1.0 / (sigma_obs[mask] ** 2 + 1e-12)
            mu_inst = float(np.sum(w * rv_obs[mask]) / np.sum(w))
            mu_per_inst[str(inst)] = mu_inst
            model[mask] = mu_inst
        mu = mu_per_inst
    else:
        w = 1.0 / (sigma_obs**2 + 1e-12)
        mu = float(np.sum(w * rv_obs) / np.sum(w))
        model[:] = mu
        k = 1
    ll = loglike_white_jitter(rv_obs, model, sigma_obs, 0.0)
    n = rv_obs.shape[0]
    bic = -2.0 * ll + k * np.log(n)
    return {"mu": mu, "ll": ll, "bic": bic}

def bic_from_ll(ll: float, k_params: int, n_points: int) -> float:
    return float(-2.0*ll + k_params * np.log(n_points))

def summarize_residuals(rv_obs: np.ndarray, rv_model: np.ndarray) -> Dict[str,float]:
    resid = rv_obs - rv_model
    rms = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    return {"rms": rms, "mae": mae}

def parameter_matching_score(
    config: SystemConfig,
    truth: List[PlanetParams],
    guesses: List[PlanetParams],
    times_days: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    Mstar = config.star.M_star_sun
    match = match_planets(truth, guesses, M_star_sun=Mstar, times_days=times_days)
    pairs = match["pairs"]
    if len(pairs)==0 and (len(truth)>0 or len(guesses)>0):
        score = 0.0
    else:
        s_list = [math.exp(-d) for (_,_,d) in pairs]
        score = float(np.mean(s_list)) if len(s_list)>0 else 1.0
        # For the strictest pass criterion (every planet must individually clear
        # the threshold rather than only the mean), swap the line above for:
        #     score = float(np.min(s_list)) if len(s_list)>0 else 1.0
    count_penalty = -0.25 * (abs(len(truth) - len(guesses)))
    return {"score": score + count_penalty, "assignment": match}

def assemble_reward(components: Dict[str, float], weights: Dict[str, float]) -> float:
    r = 0.0
    for k, w in weights.items():
        r += w * components.get(k, 0.0)
    return float(r)


def _parse_submission_planets(submission: Dict[str, Any], mode: str) -> List[PlanetParams]:
    if mode not in ("params_and_model", "params_only"):
        return []
    if "planets" not in submission:
        raise ValueError(f"Submission must include 'planets' in mode '{mode}'")
    if not isinstance(submission["planets"], list):
        raise TypeError("Submission 'planets' must be a list of dicts")
    guesses: List[PlanetParams] = []
    for pd in submission["planets"]:
        if "P_days" not in pd:
            raise ValueError("Each planet must include 'P_days'")
        guesses.append(
            PlanetParams(
                P_days=float(pd["P_days"]),
                m_sin_i_mjup=float(pd.get("m_sin_i_mjup", 0.1)),
                e=float(pd.get("e", 0.0)),
                inc_rad=float(pd.get("inc_rad", 0.0)),
                Omega_rad=float(pd.get("Omega_rad", 0.0)),
                omega_rad=float(pd.get("omega_rad", 0.0)),
                l_rad=float(pd.get("l_rad", 0.0)),
                m_true_mjup=None,
            )
        )
    return guesses


def _mle_gamma_offset(
    rv_obs: np.ndarray,
    rv_planet_only: np.ndarray,
    sigma_obs: np.ndarray,
    sigma_jitter: float,
) -> float:
    var = sigma_obs**2 + sigma_jitter**2 + 1e-12
    w = 1.0 / var
    return float(np.sum(w * (rv_obs - rv_planet_only)) / np.sum(w))

def evaluate_submission(
    config: SystemConfig,
    obs: Observations,
    submission: Dict[str, Any],
    truth_planets: List[PlanetParams],
    reward_weights: Dict[str, float],
    mode: str = "params_and_model"
) -> Tuple[float, Dict[str, Any]]:
    """Compute reward and detailed metrics.

    Args:
        mode: one of {"params_and_model","model_only","params_only"}
    """
    if not isinstance(submission, dict):
        raise TypeError(f"Submission must be a dictionary, got {type(submission).__name__}")
    VALID_MODES = {"params_and_model","model_only","params_only"}
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Valid modes: {VALID_MODES}")

    t = np.array(obs.times_days, dtype=float)
    y = np.array(obs.rvs_ms, dtype=float)
    s = np.array(obs.sigmas_ms, dtype=float)
    n = y.shape[0]
    info: Dict[str, Any] = {}
    components: Dict[str, float] = {}

    guesses = _parse_submission_planets(submission, mode)

    inst_arr = np.array(obs.instruments) if obs.instruments else None
    unique_inst = np.unique(inst_arr) if inst_arr is not None else np.array(["default"])
    n_inst = len(unique_inst)

    if mode in ("params_and_model", "model_only"):
        sigma_jit = float(submission.get("noise", {}).get("sigma_jitter_ms", 0.0))
        if mode == "params_and_model":
            rv_planet = simulate_rv_keplerian(
                planets=guesses,
                times_days=t,
                M_star_sun=float(config.star.M_star_sun),
                gamma_ms=0.0,
            )
            # Per-instrument gamma offset fitting
            rv_model = rv_planet.copy()
            gamma_per_inst: Dict[str, float] = {}
            for inst in unique_inst:
                mask = inst_arr == inst
                gamma_inst = _mle_gamma_offset(y[mask], rv_planet[mask], s[mask], sigma_jit)
                gamma_per_inst[str(inst)] = gamma_inst
                rv_model[mask] += gamma_inst
            info["rv_model_source"] = "rv_only_keplerian_from_planets"
            info["gamma_per_instrument_ms"] = gamma_per_inst
        else:
            if "rv_model" not in submission:
                raise ValueError(f"Submission must include 'rv_model' in mode '{mode}'")
            rv_model = np.array(submission["rv_model"], dtype=float)
            if rv_model.ndim != 1:
                raise ValueError(f"rv_model must be 1D. Got shape {rv_model.shape}")
            if rv_model.shape[0] != y.shape[0]:
                raise ValueError(f"rv_model length mismatch: expected {y.shape[0]}, got {rv_model.shape[0]}")
            info["rv_model_source"] = "submission.rv_model"

        ll = loglike_white_jitter(y, rv_model, s, sigma_jit)
        k = len(guesses) * 5 + n_inst  # 5 orbital params per planet + 1 gamma per instrument
        bic = bic_from_ll(ll, k_params=k, n_points=n)
        null = best_constant_fit(y, s, instruments=inst_arr)
        delta_bic = null["bic"] - bic
        resid_stats = summarize_residuals(y, rv_model)
        info.update({"ll": ll, "bic": bic, "delta_bic": delta_bic, "residuals": resid_stats, "bic_null": null["bic"]})
        components["likelihood"] = ll / n
        components["delta_bic"] = delta_bic / n
        components["neg_rms"] = -resid_stats["rms"]

    if mode in ("params_and_model", "params_only"):
        match = parameter_matching_score(config, truth_planets, guesses, times_days=t)
        info["matching"] = match
        components["match"] = match["score"]
        components["count"] = -abs(len(truth_planets) - len(guesses))

    reward = assemble_reward(components, reward_weights)
    info["components"] = components
    info["mode"] = mode
    return reward, info
