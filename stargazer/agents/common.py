from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stargazer.config import (
    GPParams,
    InstrumentParams,
    NoiseParams,
    ObservingSchedule,
    PlanetParams,
    StarParams,
    SystemConfig,
)
from stargazer.engine_rebound import simulate_clean_rv
from stargazer.utils_units import semi_amplitude_ms

class LLMActionFormatError(RuntimeError):
    """Raised when an LLM output cannot be transformed into a valid Stargazer action."""


def _coerce_float(value: Any, *, name: str, default: Optional[float] = None) -> float:
    """Best-effort conversion of LLM-provided values into finite floats.

    LLMs sometimes emit JSON nulls (parsed as Python None). Treat those as "missing"
    when a default is provided.
    """
    if value is None:
        if default is None:
            raise LLMActionFormatError(f"`{name}` must be a real number, got null.")
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise LLMActionFormatError(f"`{name}` must be a real number, got {value!r}.") from exc
    if not np.isfinite(out):
        raise LLMActionFormatError(f"`{name}` must be finite, got {out}.")
    return out


def _infer_star_mass_sun(observation: Dict[str, Any]) -> Optional[float]:
    meta = observation.get("meta")
    if not isinstance(meta, dict):
        return None
    val = meta.get("star_mass_sun")
    if val is None:
        return None
    return _coerce_float(val, name="meta.star_mass_sun")


def _infer_los_axis(observation: Dict[str, Any]) -> str:
    meta = observation.get("meta")
    if isinstance(meta, dict):
        axis = meta.get("los_axis")
        if axis in ("x", "y", "z"):
            return str(axis)
    return "x"


def _infer_integrator_preference(observation: Dict[str, Any]) -> str:
    meta = observation.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("integrator_preference"), str):
        return str(meta["integrator_preference"])
    return "whfast"


def _mass_from_semi_amplitude(
    semi_amplitude_ms: float,
    period_days: float,
    eccentricity: float,
    star_mass_sun: float,
) -> float:
    # Invert stargazer.utils_units.semi_amplitude_ms (Perryman 2018 approximation).
    if period_days <= 0:
        raise LLMActionFormatError(f"Period must be >0 days, got {period_days}")
    if star_mass_sun <= 0:
        raise LLMActionFormatError(f"Stellar mass must be >0, got {star_mass_sun}")
    e = float(np.clip(eccentricity, 0.0, 0.999))
    P_years = period_days / 365.25
    denom = (
        28.4329
        * (star_mass_sun ** (-2.0 / 3.0))
        * (P_years ** (-1.0 / 3.0))
        / np.sqrt(1.0 - e * e)
    )
    if denom <= 0 or not np.isfinite(denom):
        raise LLMActionFormatError("Invalid semi-amplitude inversion.")
    return float(semi_amplitude_ms / denom)


def _parse_phase_to_l_rad(
    desc: Dict[str, Any],
    omega_rad: float,
    Omega_rad: float,
    period: float,
    t0: float,
) -> float:
    """Parse various phase representations and convert to l_rad (mean longitude).

    Canonical rule:
    1. If explicit `l_rad` is present, always use it.
    2. Otherwise infer from legacy phase fields using this order:
       phase_frac, phase_rad/phase_deg/phase, T0_days/T_peri.

    This keeps parser behavior consistent with semantic validation and scoring:
    when `l_rad` is provided, legacy aliases are treated as non-canonical hints.

    Legacy priority (when `l_rad` is absent):
    1. phase_frac (common agent output, interpreted as time-of-periastron fraction)
    2. phase_rad / phase_deg / phase (interpreted as mean anomaly M0)
    3. T0_days / T_peri (time of periastron)
    4. Default to 0.0
    """
    
    def _get_float(key: str) -> Optional[float]:
        if key in desc:
            return _coerce_float(desc.get(key), name=key, default=None)
        return None
    
    # Collect all phase-related values
    l_rad_val = _get_float("l_rad")
    phase_frac_val = _get_float("phase_frac")
    phase_rad_val = _get_float("phase_rad")
    phase_deg_val = _get_float("phase_deg")
    phase_val = _get_float("phase")
    t0_days_val = _get_float("T0_days") or _get_float("T_peri")
    
    # Helper: convert phase_frac to l_rad
    # Agent convention: tp = t0 + phase_frac * P (time of periastron)
    # At t=t0, M = -2π * phase_frac, so l = Omega + omega + M
    def l_from_phase_frac(pf: float) -> float:
        M0 = (-2.0 * np.pi * pf) % (2.0 * np.pi)
        return (Omega_rad + omega_rad + M0) % (2.0 * np.pi)
    
    # Helper: convert M0 (mean anomaly) to l_rad
    def l_from_M0(M0: float) -> float:
        return (Omega_rad + omega_rad + M0) % (2.0 * np.pi)
    
    # Helper: convert time of periastron to l_rad
    def l_from_T0(T0: float) -> float:
        n = 2.0 * np.pi / period
        M0 = (-n * (T0 - t0)) % (2.0 * np.pi)
        return (Omega_rad + omega_rad + M0) % (2.0 * np.pi)
    
    # Canonical rule: explicit l_rad wins over any legacy field.
    if l_rad_val is not None:
        return l_rad_val % (2.0 * np.pi)

    candidates = []

    # No explicit l_rad: fall back to legacy fields.
    if phase_frac_val is not None and abs(phase_frac_val) > 1e-9:
        candidates.append(("phase_frac", l_from_phase_frac(phase_frac_val)))
    
    # Check phase_rad (as M0)
    if phase_rad_val is not None and abs(phase_rad_val) > 1e-9:
        candidates.append(("phase_rad", l_from_M0(phase_rad_val)))
    
    # Check phase_deg (as M0)
    if phase_deg_val is not None and abs(phase_deg_val) > 1e-9:
        M0_deg = float(np.deg2rad(phase_deg_val))
        candidates.append(("phase_deg", l_from_M0(M0_deg)))
    
    # Check generic phase (heuristic for radians vs degrees)
    if phase_val is not None and abs(phase_val) > 1e-9:
        if abs(phase_val) <= 2.0 * np.pi + 1e-6:
            M0_phase = phase_val
        elif abs(phase_val) <= 360.0 + 1e-6:
            M0_phase = float(np.deg2rad(phase_val))
        else:
            M0_phase = phase_val
        candidates.append(("phase", l_from_M0(M0_phase)))
    
    # Check time of periastron
    if t0_days_val is not None:
        candidates.append(("T0_days", l_from_T0(t0_days_val)))
    
    # If we found non-zero candidates, use the first one (highest priority)
    if candidates:
        return candidates[0][1]
    
    # Fallback: check for zero-valued fields (in case agent explicitly set them to 0)
    if l_rad_val is not None:
        return l_rad_val % (2.0 * np.pi)
    if phase_frac_val is not None:
        return l_from_phase_frac(phase_frac_val)
    if phase_rad_val is not None:
        return l_from_M0(phase_rad_val)
    if phase_deg_val is not None:
        return l_from_M0(float(np.deg2rad(phase_deg_val)))
    if phase_val is not None:
        if abs(phase_val) <= 2.0 * np.pi + 1e-6:
            return l_from_M0(phase_val)
        elif abs(phase_val) <= 360.0 + 1e-6:
            return l_from_M0(float(np.deg2rad(phase_val)))
        else:
            return l_from_M0(phase_val)
    
    # Default
    return 0.0


def plan_to_action(
    plan: Dict[str, Any],
    observation: Dict[str, Any],
    max_planets: int,
    submission_mode: str,
) -> Dict[str, Any]:
    """Convert a JSON-like planet plan into an Stargazer action."""
    times = np.asarray(observation["times_days"], dtype=float)
    rvs = np.asarray(observation["rvs_ms"], dtype=float)
    sigmas = np.asarray(observation["sigmas_ms"], dtype=float)
    weights = 1.0 / np.maximum(sigmas**2, 1e-6)
    default_offset = float(np.average(rvs, weights=weights)) if rvs.size else 0.0
    offset = _coerce_float(plan.get("rv_offset_ms"), name="rv_offset_ms", default=default_offset)

    planets_desc = plan.get("planets", [])
    if not isinstance(planets_desc, list):
        raise LLMActionFormatError("`planets` field must be a list.")

    model = np.full_like(rvs, offset)
    planets: List[Dict[str, Any]] = []
    star_mass_sun = _infer_star_mass_sun(observation)
    for desc in planets_desc[:max_planets]:
        component, planet = planet_from_desc(desc, times, star_mass_sun=star_mass_sun)
        model += component
        planets.append(planet)

    if submission_mode in {"params_and_model", "model_only"} and planets:
        def _keplerian_rv_from_planets(
            planets_params: List[PlanetParams],
            times_days: np.ndarray,
            star_mass_sun_val: float,
        ) -> np.ndarray:
            """Compute a non-interacting Keplerian RV model (no REBOUND dependency).

            This uses standard RV convention:
              rv(t) = Σ K [cos(ν(t)+ω) + e cos ω]

            Mean anomaly is derived from the provided mean longitude `l_rad` (λ0) at `times_days[0]`:
              M0 = λ0 - (Ω + ω)
            """

            if times_days.size == 0:
                return np.zeros(0, dtype=float)

            t_ref = float(times_days[0])
            rv_total = np.zeros(times_days.shape, dtype=float)

            def _wrap(x: np.ndarray) -> np.ndarray:
                return np.mod(x, 2.0 * np.pi)

            def _solve_kepler(M: np.ndarray, ecc: float) -> np.ndarray:
                # Newton solve for eccentric anomaly E: E - e sin E = M
                if ecc == 0.0:
                    return _wrap(M)
                e = float(np.clip(ecc, 0.0, 0.999999))
                E = _wrap(M + e * np.sin(M))
                for _ in range(80):
                    f = E - e * np.sin(E) - M
                    fp = 1.0 - e * np.cos(E)
                    step = f / fp
                    E = E - step
                    if float(np.max(np.abs(step))) < 1e-12:
                        break
                return _wrap(E)

            for p in planets_params:
                P = float(p.P_days)
                if P <= 0:
                    continue
                e = float(np.clip(p.e, 0.0, 0.8))
                omega = float(p.omega_rad)
                Omega = float(p.Omega_rad)
                lambda0 = float(p.l_rad)
                M0 = (lambda0 - (Omega + omega)) % (2.0 * np.pi)

                n = 2.0 * np.pi / P
                M = _wrap(M0 + n * (times_days - t_ref))
                E = _solve_kepler(M, e)
                nu = 2.0 * np.arctan2(
                    np.sqrt(1.0 + e) * np.sin(E / 2.0),
                    np.sqrt(1.0 - e) * np.cos(E / 2.0),
                )

                K = float(
                    semi_amplitude_ms(
                        float(p.m_sin_i_mjup),
                        float(p.P_days),
                        float(p.e),
                        float(star_mass_sun_val),
                    )
                )
                rv_total += K * (np.cos(nu + omega) + e * np.cos(omega))

            return rv_total

        try:
            planets_params = [PlanetParams(**p) for p in planets]
            # Convert RV-only params (Omega=0) to REBOUND geometry.
            # The Keplerian model's LOS convention differs from REBOUND's
            # coordinate axes.  For los_axis='x' the orbit needs Omega=π/2;
            # for 'y' it needs Omega=π.  l_rad shifts by the same amount
            # so that M0 is preserved.
            los_axis = _infer_los_axis(observation)
            _omega_shift = {"x": np.pi / 2.0, "y": np.pi, "z": 0.0}
            dOmega = _omega_shift.get(los_axis, np.pi / 2.0)
            planets_rebound = [
                PlanetParams(
                    P_days=p.P_days,
                    m_sin_i_mjup=p.m_sin_i_mjup,
                    e=p.e,
                    inc_rad=p.inc_rad,
                    Omega_rad=(p.Omega_rad + dOmega) % (2.0 * np.pi),
                    omega_rad=p.omega_rad,
                    l_rad=(p.l_rad + dOmega) % (2.0 * np.pi),
                    m_true_mjup=p.m_true_mjup,
                )
                for p in planets_params
            ]
            rv_clean = simulate_clean_rv(
                SystemConfig(
                    star=StarParams(M_star_sun=float(star_mass_sun or 1.0), gamma_ms=0.0),
                    planets=planets_rebound,
                    schedule=ObservingSchedule(times_days=times.tolist(), instruments=["instA"] * int(times.size)),
                    instruments=[
                        InstrumentParams(label="instA", gamma_ms=0.0, sigma_white_ms=1.0, sigma_jitter_ms=0.0)
                    ],
                    noise=NoiseParams(sigma_white_ms=1.0, sigma_jitter_ms=0.0, gp=GPParams(use_gp=False)),
                    engine="rebound",
                    los_axis=los_axis,
                    integrator_preference=_infer_integrator_preference(observation),
                ),
                times.tolist(),
                t_ref_days=float(times[0]),
            )
            model = rv_clean + offset
        except Exception:
            # Fall back to an analytic Keplerian RV model (still uses the submitted planet params).
            # This prevents large mismatches when REBOUND is unavailable or integration fails.
            planets_params = [PlanetParams(**p) for p in planets]
            model = _keplerian_rv_from_planets(
                planets_params,
                times,
                float(star_mass_sun or 1.0),
            ) + offset

    default_jitter = float(np.sqrt(np.average((rvs - model) ** 2, weights=weights))) if rvs.size else 0.0
    jitter = _coerce_float(plan.get("noise_jitter_ms"), name="noise_jitter_ms", default=default_jitter)
    action = {
        "rv_model": model.tolist(),
        "planets": planets,
        "noise": {"sigma_jitter_ms": float(max(0.1, jitter))},
    }

    if submission_mode == "model_only":
        return {key: action[key] for key in ("rv_model", "noise") if key in action}
    if submission_mode == "params_only":
        return {"planets": planets}
    return action


def canonicalize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a plan in-place to avoid conflicting aliases."""
    if not isinstance(plan, dict):
        return plan
    planets_desc = plan.get("planets", [])
    if not isinstance(planets_desc, list):
        return plan
    cleaned = []
    for desc in planets_desc:
        if not isinstance(desc, dict):
            cleaned.append(desc)
            continue
        out = dict(desc)
        if "P_days" not in out and "period_days" in out:
            out["P_days"] = out["period_days"]
        if "e" not in out and "eccentricity" in out:
            out["e"] = out["eccentricity"]
        if "P_days" in out and "period_days" in out:
            out.pop("period_days", None)
        if "e" in out and "eccentricity" in out:
            out.pop("eccentricity", None)
        # Prefer Stargazer-native mass parameterization when both are present.
        if "m_sin_i_mjup" in out:
            out.pop("semi_amplitude_ms", None)
            out.pop("K_ms", None)
        if "l_rad" in out:
            for k in ("phase_rad", "phase_deg", "phase", "phase_frac", "T0_days", "T_peri"):
                out.pop(k, None)
        cleaned.append(out)
    plan = dict(plan)
    plan["planets"] = cleaned
    return plan


def planet_from_desc(
    desc: Dict[str, Any],
    times: np.ndarray,
    *,
    star_mass_sun: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Convert a single planet description into a sinusoidal component and Stargazer planet entry."""
    if not isinstance(desc, dict):
        raise LLMActionFormatError("Planet description must be an object.")

    period = _coerce_float(desc.get("period_days", desc.get("P_days")), name="period_days/P_days")
    if period <= 0.5:
        raise LLMActionFormatError(f"Period {period} days is too short.")

    # === Eccentricity ===
    ecc_raw = desc.get("eccentricity", desc.get("e"))
    ecc = float(np.clip(_coerce_float(ecc_raw, name="eccentricity/e", default=0.0), 0.0, 0.8))

    # === Argument of periastron (omega) ===
    omega_rad = _coerce_float(desc.get("omega_rad"), name="omega_rad", default=0.0)

    # === 3D orientation (used by REBOUND forward model) ===
    # Keep backward-compatible defaults if omitted.
    inc_rad = _coerce_float(desc.get("inc_rad"), name="inc_rad", default=float(np.pi / 2.0))
    inc_rad = float(np.clip(inc_rad, 0.0, np.pi))
    Omega_rad = _coerce_float(desc.get("Omega_rad"), name="Omega_rad", default=0.0)
    Omega_rad = float(Omega_rad % (2.0 * np.pi))
    
    # === Semi-amplitude / Mass ===
    amp: float
    if "semi_amplitude_ms" in desc or "K_ms" in desc:
        amp = _coerce_float(
            desc.get("semi_amplitude_ms", desc.get("K_ms")),
            name="semi_amplitude_ms/K_ms",
            default=0.0,
        )
    elif "m_sin_i_mjup" in desc:
        # If the caller provides Stargazer-style parameters, approximate K assuming M_star=1 Msun.
        P_years = period / 365.25
        amp = float(
            28.4329
            * _coerce_float(desc.get("m_sin_i_mjup"), name="m_sin_i_mjup", default=0.0)
            * (P_years ** (-1.0 / 3.0))
        )
    else:
        amp = 0.0
    amp = float(np.clip(amp, 0.0, 200.0))  # Increased max to 200 m/s

    # === Phase / Mean longitude ===
    t0 = float(times[0]) if times.size > 0 else 0.0
    l_rad = _parse_phase_to_l_rad(desc, omega_rad, Omega_rad, period, t0)

    # === Compute sinusoidal approximation (fallback) ===
    ang_freq = 2.0 * np.pi / period
    # For sinusoidal approximation, use l_rad as phase
    # Note: This is a circular orbit approximation; REBOUND will compute the real RV
    component = amp * np.sin(ang_freq * (times - t0) + l_rad)

    # === Compute mass from amplitude ===
    P_years = period / 365.25
    if "m_sin_i_mjup" in desc and _coerce_float(desc.get("m_sin_i_mjup"), name="m_sin_i_mjup", default=0.0) > 0:
        mass = _coerce_float(desc.get("m_sin_i_mjup"), name="m_sin_i_mjup", default=0.1)
    else:
        if star_mass_sun is not None and star_mass_sun > 0:
            mass = _mass_from_semi_amplitude(amp, period, ecc, float(star_mass_sun))
        else:
            # Fallback: assume 1 solar mass
            mass = amp / (28.4329 * (P_years ** (-1.0 / 3.0)))
    
    # === Build output planet dict ===
    planet = {
        "P_days": float(period),
        "m_sin_i_mjup": float(np.clip(mass, 1e-3, 30.0)),
        "e": ecc,
        "inc_rad": inc_rad,
        "Omega_rad": Omega_rad,
        "omega_rad": omega_rad % (2 * np.pi),
        "l_rad": l_rad,
    }
    return component, planet


def validate_submission_semantics(
    plan: Dict[str, Any],
    observation: Dict[str, Any],
    *,
    phase_conflict_tol_rad: float = 0.35,
) -> Dict[str, Any]:
    """Validate protocol-level parameter semantics before submission.

    Focuses on:
    - phase semantic consistency (`l_rad` vs `phase_*`/`T0_days`)
    - reference epoch usage (`t_ref = times_days[0]`)
    - legacy alias usage visibility
    """

    def _ang_diff(a: float, b: float) -> float:
        d = (a - b + np.pi) % (2.0 * np.pi) - np.pi
        return float(abs(d))

    def _try_float(desc: Dict[str, Any], key: str) -> Optional[float]:
        if key not in desc:
            return None
        try:
            return _coerce_float(desc.get(key), name=key, default=None)
        except LLMActionFormatError:
            return None

    times = np.asarray(observation.get("times_days", []), dtype=float)
    t_ref = float(times[0]) if times.size else 0.0
    planets_desc = plan.get("planets", [])

    out: Dict[str, Any] = {
        "ok": True,
        "t_ref_days": t_ref,
        "errors": [],
        "warnings": [],
        "per_planet": [],
    }

    if not isinstance(planets_desc, list):
        out["ok"] = False
        out["errors"].append("`planets` must be a list.")
        return out

    legacy_keys = {
        "period_days",
        "semi_amplitude_ms",
        "eccentricity",
        "phase_rad",
        "phase_deg",
        "phase",
        "phase_frac",
        "K_ms",
        "T0_days",
        "T_peri",
    }

    for i, desc in enumerate(planets_desc):
        if not isinstance(desc, dict):
            out["ok"] = False
            out["errors"].append(f"planet[{i}] must be an object.")
            continue

        period = _try_float(desc, "period_days")
        if period is None:
            period = _try_float(desc, "P_days")
        omega = _try_float(desc, "omega_rad")
        omega = float(omega) if omega is not None else 0.0
        Omega = _try_float(desc, "Omega_rad")
        Omega = float(Omega) if Omega is not None else 0.0

        if period is None or period <= 0.5:
            out["ok"] = False
            out["errors"].append(f"planet[{i}] invalid period; expected `P_days` > 0.5.")
            continue

        candidates: Dict[str, float] = {}
        l_direct = _try_float(desc, "l_rad")
        if l_direct is not None:
            candidates["l_rad"] = float(l_direct % (2.0 * np.pi))

        phase_rad = _try_float(desc, "phase_rad")
        if phase_rad is not None:
            candidates["phase_rad(M0)"] = float((Omega + omega + phase_rad) % (2.0 * np.pi))

        phase_deg = _try_float(desc, "phase_deg")
        if phase_deg is not None:
            candidates["phase_deg(M0)"] = float((Omega + omega + np.deg2rad(phase_deg)) % (2.0 * np.pi))

        phase = _try_float(desc, "phase")
        if phase is not None:
            if abs(phase) <= 2.0 * np.pi + 1e-6:
                phase_as_rad = phase
            elif abs(phase) <= 360.0 + 1e-6:
                phase_as_rad = float(np.deg2rad(phase))
            else:
                phase_as_rad = phase
            candidates["phase(alias M0)"] = float((Omega + omega + phase_as_rad) % (2.0 * np.pi))

        phase_frac = _try_float(desc, "phase_frac")
        if phase_frac is not None:
            M0 = (-2.0 * np.pi * phase_frac) % (2.0 * np.pi)
            candidates["phase_frac(tp)"] = float((Omega + omega + M0) % (2.0 * np.pi))

        T0 = _try_float(desc, "T0_days")
        if T0 is None:
            T0 = _try_float(desc, "T_peri")
        if T0 is not None:
            n = 2.0 * np.pi / float(period)
            M0 = (-n * (T0 - t_ref)) % (2.0 * np.pi)
            candidates["T0_days(tp)"] = float((Omega + omega + M0) % (2.0 * np.pi))

        planet_info: Dict[str, Any] = {
            "planet_idx": i,
            "t_ref_days": t_ref,
            "candidate_l_rad": candidates,
            "resolved_l_rad": float(_parse_phase_to_l_rad(desc, omega, Omega, float(period), t_ref)),
        }

        keys = list(candidates.keys())
        if len(keys) >= 2:
            max_diff = 0.0
            worst_pair = None
            for a_idx in range(len(keys)):
                for b_idx in range(a_idx + 1, len(keys)):
                    ka, kb = keys[a_idx], keys[b_idx]
                    d = _ang_diff(candidates[ka], candidates[kb])
                    if d > max_diff:
                        max_diff = d
                        worst_pair = (ka, kb, d)
            planet_info["max_phase_disagreement_rad"] = max_diff
            if max_diff > phase_conflict_tol_rad and worst_pair is not None:
                # If explicit l_rad is provided, treat it as canonical and do not hard-fail.
                # Many agents keep legacy phase_* fields as placeholders (often 0.0).
                if "l_rad" in candidates:
                    out["warnings"].append(
                        "planet[{i}] phase fields disagree ({a} vs {b}, Δ={d:.3f} rad), "
                        "but `l_rad` is present and treated as canonical.".format(
                            i=i, a=worst_pair[0], b=worst_pair[1], d=worst_pair[2]
                        )
                    )
                else:
                    out["ok"] = False
                    out["errors"].append(
                        "planet[{i}] phase semantics conflict: {a} vs {b} differ by {d:.3f} rad. "
                        "Use a single convention or make them consistent.".format(
                            i=i, a=worst_pair[0], b=worst_pair[1], d=worst_pair[2]
                        )
                    )

        used_legacy = sorted(k for k in desc.keys() if k in legacy_keys)
        if used_legacy:
            out["warnings"].append(
                f"planet[{i}] used legacy aliases {used_legacy}; prefer Stargazer native keys "
                f"(P_days, m_sin_i_mjup, e, omega_rad, l_rad)."
            )
        if not any(k in desc for k in ("l_rad", "phase_rad", "phase_deg", "phase", "phase_frac", "T0_days", "T_peri")):
            out["warnings"].append(
                f"planet[{i}] has no phase field; defaulting to l_rad=0 at t_ref={t_ref}."
            )

        out["per_planet"].append(planet_info)

    return out
