from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import numpy as np, uuid

from dataclasses import asdict
from .config import *
from .priors import sample_planets, sample_noise_params
from .schedule import random_uniform_over_baseline
from .engine_rebound import simulate_clean_rv
from .noise import draw_gp_process
from .utils_time import baseline_days
from .utils_units import semi_amplitude_ms

class TaskFactory:
    """
    Factory class for generating Stargazer tasks with configurable difficulty.

    This provides a convenient interface for sampling tasks, especially useful
    when creating task samplers for environments.
    """
    def __init__(self,
                 difficulty: str = "auto",
                 multiplicity_probs=(0.5, 0.3, 0.15, 0.05),
                 resonance_fraction: float = 0.3,
                 n_obs_range=(40, 80),
                 engine: str = "rebound",
                 los_axis: str = "x",
                 integrator_preference: str = "whfast"):
        """
        Initialize the TaskFactory.

        Args:
            difficulty: Difficulty level (currently unused, kept for API compatibility)
            multiplicity_probs: Probabilities for 1, 2, 3, 4 planet systems
            resonance_fraction: Fraction of systems in resonance
            n_obs_range: Range for number of observations
            engine: Physics engine ("rebound" or "keplerian")
            los_axis: Line-of-sight axis ("x", "y", or "z")
            integrator_preference: Integration method for rebound
        """
        self.difficulty = difficulty
        self.multiplicity_probs = multiplicity_probs
        self.resonance_fraction = resonance_fraction
        self.n_obs_range = n_obs_range
        self.engine = engine
        self.los_axis = los_axis
        self.integrator_preference = integrator_preference
        self._seed_counter = 0

    def sample(self, seed: Optional[int] = None) -> Task:
        """
        Sample a new task with the configured parameters.

        Args:
            seed: Random seed (if None, uses auto-incrementing counter)

        Returns:
            Task: A new Stargazer task
        """
        if seed is None:
            seed = self._seed_counter
            self._seed_counter += 1

        return generate_task(
            seed=seed,
            difficulty=self.difficulty,
            multiplicity_probs=self.multiplicity_probs,
            resonance_fraction=self.resonance_fraction,
            n_obs_range=self.n_obs_range,
            engine=self.engine,
            los_axis=self.los_axis,
            integrator_preference=self.integrator_preference
        )


def evaluate_task_difficulty(config: SystemConfig,
                             observations: Observations,
                             noise: NoiseParams,
                             resonance_tolerance: float = 0.03) -> Tuple[int, Dict[str, Any]]:
    npl = len(config.planets)
    base = min(4, npl)
    Ks = [semi_amplitude_ms(p.m_sin_i_mjup, p.P_days, p.e, config.star.M_star_sun) for p in config.planets] if npl>0 else [0.0]
    minK = min(Ks) if Ks else 0.0
    denom = max(1e-6, noise.sigma_white_ms + noise.sigma_jitter_ms)
    snr = minK / denom
    snr_term = 0 if snr > 5 else (1 if snr > 2 else 2 if snr > 1 else 3)

    Ps = sorted([p.P_days for p in config.planets])
    near_res = 0
    for i in range(len(Ps)-1):
        r = Ps[i+1]/Ps[i]
        for p,q in [(2,1),(3,2),(5,3)]:
            target = p/q
            if abs(r/target - 1.0) < resonance_tolerance:
                near_res += 1
                break
    res_term = min(2, near_res)

    n_obs = len(observations.times_days)
    base_cov = baseline_days(observations.times_days) / (Ps[0] if Ps else 1.0)
    cov_term = 0 if base_cov >= 3 else (1 if base_cov >= 2 else 2)
    nobs_term = 0 if n_obs >= 80 else (1 if n_obs >= 50 else 2 if n_obs >= 30 else 3)

    red_term = 0
    if noise.gp.use_gp:
        red_amp = noise.gp.sigma_ms / max(1e-6, noise.sigma_white_ms)
        red_term = 1 if red_amp < 0.5 else 2 if red_amp < 1.0 else 3

    score = base + snr_term + res_term + cov_term + nobs_term + red_term
    level = max(1, min(10, int(round(score))))
    details = {
        "n_planets": npl, "snr_like": snr, "minK_ms": minK, "resonances_detected": near_res,
        "coverage_inner_periods": base_cov, "n_obs": n_obs, "gp_used": noise.gp.use_gp, "raw_score": score
    }
    return level, details

def generate_task(seed: Optional[int] = None,
                  difficulty: str = "auto",
                  multiplicity_probs = (0.5, 0.3, 0.15, 0.05),
                  resonance_fraction: float = 0.3,
                  n_obs_range=(40,80),
                  engine: str = "rebound",
                  los_axis: str = "x",
                  integrator_preference: str = "whfast") -> Task:
    if not isinstance(multiplicity_probs, (tuple, list)) or len(multiplicity_probs) != 4:
        raise ValueError("multiplicity_probs must be a tuple/list of length 4")
    if not np.isclose(sum(multiplicity_probs), 1.0):
        raise ValueError(f"multiplicity_probs must sum to 1.0, got {sum(multiplicity_probs)}")
    if not (0.0 <= resonance_fraction <= 1.0):
        raise ValueError(f"resonance_fraction must be in [0,1], got {resonance_fraction}")
    if not isinstance(n_obs_range, (tuple, list)) or len(n_obs_range) != 2:
        raise ValueError("n_obs_range must be a tuple/list of length 2")
    if n_obs_range[0] < 1 or n_obs_range[0] > n_obs_range[1]:
        raise ValueError(f"Invalid n_obs_range: {n_obs_range}")
    if engine not in {"rebound","keplerian"}:
        raise ValueError(f"Unknown engine '{engine}'")
    if (los_axis or 'x').lower() not in {'x','y','z'}:
        raise ValueError(f"Invalid los_axis '{los_axis}'. Must be one of x,y,z.")

    rng = np.random.default_rng(seed)
    k = rng.choice([1,2,3,4], p=np.array(multiplicity_probs, dtype=float))
    star = StarParams(M_star_sun=float(rng.uniform(0.7, 1.3)), gamma_ms=0.0)
    task_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"stargazer|seed={seed}|difficulty={difficulty}|engine={engine}|los={los_axis}|integrator={integrator_preference}"
            f"|multiplicity={tuple(multiplicity_probs)}|res_frac={resonance_fraction}|n_obs_range={tuple(n_obs_range)}",
        )
    )
    # Retry loop to avoid pathological systems
    for attempt in range(8):
        planets = sample_planets(int(k), star=star, resonance_fraction=resonance_fraction, rng=rng)
        # Reject systems with degenerate periods (ratio < 1.1) — these are
        # physically unresolvable from RV data alone.
        if len(planets) >= 2:
            Ps = sorted(p.P_days for p in planets)
            if any(Ps[i+1] / Ps[i] < 1.1 for i in range(len(Ps) - 1)):
                if attempt < 7:
                    continue
        innerP = min([p.P_days for p in planets]) if planets else 10.0
        n_obs = int(rng.integers(n_obs_range[0], n_obs_range[1]+1))
        schedule = random_uniform_over_baseline(innerP, n_obs=n_obs, baseline_multiplier_range=(2.0,4.0), instrument_label="instA", seed=seed)
        noise = sample_noise_params(rng=rng)
        instruments = [InstrumentParams(label="instA", gamma_ms=0.0, sigma_white_ms=noise.sigma_white_ms, sigma_jitter_ms=noise.sigma_jitter_ms)]
        config = SystemConfig(star=star, planets=planets, schedule=schedule, instruments=instruments,
                              noise=noise,
                              engine=engine, los_axis=los_axis, integrator_preference=integrator_preference)
        try:
            rv_clean = simulate_clean_rv(config, schedule.times_days)
        except Exception:
            if attempt == 7:
                raise
            continue
        # instrument + systemic gamma
        inst_gamma = {inst.label: inst.gamma_ms for inst in instruments}
        gamma_series = np.array([inst_gamma.get(lbl, 0.0) + star.gamma_ms for lbl in schedule.instruments], dtype=float)
        rv_clean = rv_clean + gamma_series
        # White sigma per-point
        sigma = np.full_like(rv_clean, noise.sigma_white_ms, dtype=float)
        # GP diagonal uses white noise only; jitter added separately
        gp_draw = draw_gp_process(np.array(schedule.times_days, dtype=float), noise.gp, diag_ms2=sigma**2)
        white_plus_jitter = rng.normal(0.0, np.sqrt(sigma**2 + noise.sigma_jitter_ms**2), size=rv_clean.shape)
        rv_noisy = rv_clean + gp_draw + white_plus_jitter
        obs = Observations(times_days=schedule.times_days, rvs_ms=rv_noisy.tolist(), sigmas_ms=sigma.tolist(), instruments=schedule.instruments)
        difficulty_level, diff_details = evaluate_task_difficulty(config, obs, noise)
        task = Task(
            task_id=task_id,
            config=config,
            observations=obs,
            truth_difficulty=difficulty_level,
            difficulty_details=diff_details,
            meta={"seed": seed, "noise": asdict(noise)}
        )
        return task
    raise RuntimeError("Failed to generate a valid task after multiple attempts.")
