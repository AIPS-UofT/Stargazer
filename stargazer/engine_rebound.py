from __future__ import annotations
from typing import List, Tuple, Optional
import math, numpy as np

from .config import SystemConfig
from .utils_units import DAY_S, msun_to_kg, mjup_to_kg, days_to_seconds

def _import_rebound():
    try:
        import rebound  # type: ignore
        return rebound
    except ImportError as e:
        raise RuntimeError(
            "The 'rebound' package is required for truth RV generation but is not installed. "
            "Install with: pip install rebound"
        ) from e

def _choose_integrator(sim, preference: str):
    pref = (preference or 'whfast').lower()
    if pref in ('whfast', 'wh'):
        sim.integrator = 'whfast'
    elif pref in ('ias15',):
        sim.integrator = 'ias15'
    else:
        sim.integrator = 'whfast'  # fallback

def build_sim(config: SystemConfig):
    rebound = _import_rebound()
    sim = rebound.Simulation()
    sim.units = ["msun", "m", "s"]
    # Star
    sim.add(m=config.star.M_star_sun)
    # Planets with validation
    for p in config.planets:
        if p.P_days <= 0:
            raise ValueError(f"Invalid period P_days={p.P_days}")
        if not (0.0 <= p.e < 1.0):
            raise ValueError(f"Invalid eccentricity e={p.e}")
        P_s = p.P_days * DAY_S
        h = p.e * math.sin(p.omega_rad)
        k = p.e * math.cos(p.omega_rad)
        # Planet true mass from m*sin(i); guard against face-on
        if p.m_true_mjup is not None:
            m_true_mjup = p.m_true_mjup
        else:
            sin_i = math.sin(p.inc_rad)
            if abs(sin_i) < 1e-3:
                raise ValueError(f"sin(i) too small (|sin|<{1e-3}); cannot derive true mass safely.")
            m_true_mjup = p.m_sin_i_mjup / sin_i
            if m_true_mjup > 20.0:
                raise ValueError(f"Derived true mass {m_true_mjup:.2f} M_jup exceeds allowed max (20).")
        m_sun = mjup_to_kg(m_true_mjup) / msun_to_kg(1.0)
        # Use Pal inclination coordinates (ix, iy) to match Pal eccentricity coordinates (h, k)
        ix = math.sin(p.inc_rad / 2) * math.sin(p.Omega_rad)
        iy = math.sin(p.inc_rad / 2) * math.cos(p.Omega_rad)
        sim.add(m=m_sun, P=P_s, h=h, k=k, ix=ix, iy=iy, l=p.l_rad)
    sim.move_to_com()
    _choose_integrator(sim, config.integrator_preference)
    if sim.integrator == 'whfast' and len(config.planets) > 0:
        minP = min([pp.P_days for pp in config.planets]) * DAY_S
        sim.dt = max(minP / 50.0, 1e-3)  # floor to avoid zero dt
    return sim

def simulate_clean_rv(
    config: SystemConfig,
    times_days: List[float],
    t_ref_days: float = 0.0,
) -> np.ndarray:
    """Integrate N-body RV at each observation time.

    Parameters
    ----------
    t_ref_days : float
        Reference epoch in the same unit as *times_days*.  REBOUND t=0
        is mapped to this epoch, so ``l_rad`` in the config planets is
        interpreted as the mean longitude at *t_ref_days*.
        Default 0.0 preserves legacy behaviour (l_rad at JD/day 0).
    """
    rebound = _import_rebound()
    sim = build_sim(config)
    axis = (config.los_axis or 'x').lower()
    if axis not in ('x','y','z'):
        raise ValueError(f"Invalid LOS axis '{config.los_axis}'. Must be one of x,y,z.")
    rv = np.zeros(len(times_days), dtype=float)
    for i, tday in enumerate(times_days):
        try:
            sim.integrate((tday - t_ref_days) * DAY_S)
        except Exception as e:
            raise RuntimeError(f"REBOUND integration failed at t={tday} days (index {i}). Integrator={sim.integrator}. Error: {e}") from e
        star = sim.particles[0]
        vlos = star.vx if axis=='x' else (star.vy if axis=='y' else star.vz)
        rv[i] = vlos
    return rv
