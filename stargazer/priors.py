from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np, math
from numpy.random import Generator

from .config import PlanetParams, StarParams, GPParams, NoiseParams

def kipping_beta_eccentricity(alpha: float = 0.867, beta: float = 3.03, rng: Generator | None = None) -> float:
    rng = rng or np.random.default_rng()
    e = rng.beta(alpha, beta)
    return float(min(e, 0.9))

def isotropic_inclination_rad(rng: Generator | None = None) -> float:
    rng = rng or np.random.default_rng()
    u = rng.random()
    return float(math.acos(u))

def uniform_angle_rad(rng: Generator | None = None) -> float:
    rng = rng or np.random.default_rng()
    return float(2.0 * math.pi * rng.random())

def log_uniform(a: float, b: float, rng: Generator | None = None) -> float:
    rng = rng or np.random.default_rng()
    u = rng.uniform(np.log10(a), np.log10(b))
    return float(10**u)

def sample_periods_with_resonances(n: int,
                                   Pmin_days: float,
                                   Pmax_days: float,
                                   resonance_fraction: float = 0.25,
                                   resonance_set: List[Tuple[int,int]] = [(2,1),(3,2),(5,3)],
                                   resonance_scatter_frac: float = 0.02,
                                   min_period_ratio: float = 1.3,
                                   rng: Generator | None = None) -> List[float]:
    rng = rng or np.random.default_rng()
    P = []
    if n <= 0:
        return P
    P_inner = log_uniform(Pmin_days, Pmax_days, rng=rng)
    P.append(P_inner)
    for k in range(1, n):
        use_res = (rng.random() < resonance_fraction)
        if use_res:
            p, q = resonance_set[rng.integers(0, len(resonance_set))]
            ratio = p / q
            ratio *= rng.normal(1.0, resonance_scatter_frac)
            Pk = P[-1] * max(ratio, min_period_ratio)
        else:
            lo = max(P[-1]*min_period_ratio, Pmin_days)
            hi = Pmax_days
            if lo >= hi:
                # Can't fit in [lo, Pmax_days] range, extend hi beyond Pmax_days
                # The period will be clamped to Pmax_days later (line 57)
                hi = lo * 1.5
            Pk = log_uniform(lo, hi, rng=rng)
        P.append(Pk)
    P = [min(max(Pmin_days, x), Pmax_days) for x in P]
    P.sort()
    return P

def sample_planets(n: int,
                   star: StarParams,
                   Pmin_days: float = 2.0,
                   Pmax_days: float = 300.0,
                   resonance_fraction: float = 0.25,
                   rng: Generator | None = None) -> List[PlanetParams]:
    rng = rng or np.random.default_rng()
    Ps = sample_periods_with_resonances(n, Pmin_days, Pmax_days, resonance_fraction=resonance_fraction, rng=rng)
    planets: List[PlanetParams] = []
    for P in Ps:
        msi = log_uniform(0.01, 1.0, rng=rng)
        e = kipping_beta_eccentricity(rng=rng)
        # draw inclination but avoid pathological sin(i) ~ 0 which would explode true mass
        for _ in range(20):
            inc = isotropic_inclination_rad(rng=rng)
            if abs(math.sin(inc)) >= 0.05:
                break
        else:
            inc = math.pi/2  # fallback safe
        Omega = uniform_angle_rad(rng=rng)
        omega = uniform_angle_rad(rng=rng)
        l = uniform_angle_rad(rng=rng)
        planets.append(PlanetParams(P_days=P, m_sin_i_mjup=msi, e=e, inc_rad=inc, Omega_rad=Omega, omega_rad=omega, l_rad=l, m_true_mjup=None))
    return planets

def sample_gp_params(use_gp_prob: float = 0.3, rng: Generator | None = None) -> GPParams:
    rng = rng or np.random.default_rng()
    use_gp = (rng.random() < use_gp_prob)
    if not use_gp:
        return GPParams(use_gp=False)
    sigma = 0.5 * 10**rng.uniform(-1, 0.5)
    period = log_uniform(10.0, 45.0, rng=rng)
    Q0 = 0.5 + 10**rng.uniform(-1, 0.7)
    dQ = 10**rng.uniform(-1.3, 0.3)
    f = rng.uniform(0.1, 0.9)
    return GPParams(use_gp=True, sigma_ms=float(sigma), period_days=float(period), Q0=float(Q0), dQ=float(dQ), f=float(f))

def sample_noise_params(rng: Generator | None = None) -> NoiseParams:
    rng = rng or np.random.default_rng()
    sigma_white = 10**rng.uniform(-0.3, 0.7)
    sigma_jit = abs(rng.normal(0.0, 0.5))  # half-normal
    gp = sample_gp_params(use_gp_prob=0.4, rng=rng)
    return NoiseParams(sigma_white_ms=float(sigma_white), sigma_jitter_ms=float(sigma_jit), gp=gp)
