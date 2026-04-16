def submit_action_tool(max_planets: int, submission_mode: str):
    """Create a submit action tool definition for function calling."""
    return {
        "type": "function",
        "function": {
            "name": "submit_action",
            "description": (
                f"Submit a candidate RV model to the environment. "
                f"Preferred protocol: Stargazer-native planet fields "
                f"(P_days, m_sin_i_mjup, e, omega_rad, l_rad). "
                f"Provide up to {max_planets} planets with P_days > 0.5 "
                f"and eccentricity between 0 and 0.8. "
                f"This task currently expects submission_mode='{submission_mode}'. "
                f"The environment will evaluate your submission and return a reward with detailed metrics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "planets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "P_days": {
                                    "type": "number",
                                    "description": "Stargazer planet format: orbital period in days (alternative to period_days)."
                                },
                                "period_days": {
                                    "type": "number",
                                    "description": "Orbital period in days (must be > 0.5)"
                                },
                                "m_sin_i_mjup": {
                                    "type": "number",
                                    "description": "Stargazer planet format: minimum mass in Jupiter masses (optional; if provided, K is approximated assuming M_star=1Msun)."
                                },
                                "semi_amplitude_ms": {
                                    "type": "number",
                                    "description": "Semi-amplitude K in m/s (must be >= 0)"
                                },
                                "phase_deg": {
                                    "type": "number",
                                    "description": "Phase in degrees (optional, provide either phase_deg or phase_rad)"
                                },
                                "phase_rad": {
                                    "type": "number",
                                    "description": "Phase in radians (optional, provide either phase_deg or phase_rad)"
                                },
                                "phase": {
                                    "type": "number",
                                    "description": "Phase alias (optional). Interpreted as radians if |phase|<=2π, else degrees if |phase|<=360."
                                },
                                "phase_frac": {
                                    "type": "number",
                                    "description": "Phase as fraction of an orbit in [0,1) (optional). Converted to radians via 2π*phase_frac."
                                },
                                "l_rad": {
                                    "type": "number",
                                    "description": "Stargazer planet format: mean longitude at t_ref=times_days[0] in radians."
                                },
                                "eccentricity": {
                                    "type": "number",
                                    "description": "Eccentricity (between 0 and 0.8)"
                                },
                                "e": {
                                    "type": "number",
                                    "description": "Stargazer planet format: eccentricity (alias for eccentricity)."
                                },
                                "omega_rad": {
                                    "type": "number",
                                    "description": "Stargazer planet format: argument of periapsis in radians (optional; matching uses l_rad)."
                                },
                                "inc_rad": {
                                    "type": "number",
                                    "description": "Inclination in radians for REBOUND forward model (optional; default pi/2)."
                                },
                                "Omega_rad": {
                                    "type": "number",
                                    "description": "Longitude of ascending node in radians for REBOUND forward model (optional; default 0)."
                                }
                            },
                            "description": "Recommended: use Stargazer native fields (P_days, m_sin_i_mjup, e, omega_rad, l_rad). Legacy aliases are accepted."
                        },
                        "description": "List of planet hypotheses, sorted by confidence (most confident first)."
                    },
                    "rv_offset_ms": {
                        "type": "number",
                        "description": "Constant RV offset in m/s (optional, will be estimated if not provided)."
                    },
                    "noise_jitter_ms": {
                        "type": "number",
                        "description": "Optional white-noise jitter term in m/s."
                    },
                    "notes": {
                        "type": "string",
                        "description": "Brief rationale for this submission (optional but recommended)."
                    }
                },
                "required": ["planets"]
            }
        }
    }


def execute_submit_action(payload: dict, env, current_obs: dict, max_planets: int) -> tuple:
    """
    Execute the submit action tool with the given parameters.

    Args:
        payload: The submission payload from the LLM
        env: The Stargazer environment
        current_obs: Current observation dictionary
        max_planets: Maximum number of planets allowed

    Returns:
        tuple: (result_string, observation, reward, done, info)
    """
    from ..common import plan_to_action, validate_submission_semantics, LLMActionFormatError, canonicalize_plan
    import json

    payload = canonicalize_plan(payload)
    semantic_check = validate_submission_semantics(payload, current_obs)
    if not bool(semantic_check.get("ok", True)):
        return (
            "Plan rejected by semantic validator:\n"
            + json.dumps(
                {
                    "errors": semantic_check.get("errors", []),
                    "t_ref_days": semantic_check.get("t_ref_days"),
                    "hint": "Use Stargazer-native keys and ensure phase semantics are consistent.",
                },
                indent=2,
            ),
            current_obs,
            0.0,
            False,
            {},
        )

    try:
        action = plan_to_action(
            payload,
            current_obs,
            max_planets=max_planets,
            submission_mode=env.submission_mode,
        )
    except LLMActionFormatError as exc:
        return f"Plan rejected: {exc}", current_obs, 0.0, False, {}

    obs_next, reward, done, info = env.step(action)

    # Build a comprehensive result message
    metrics = info.get("metrics", {})
    comparison = None
    try:
        task = env.task  # type: ignore[attr-defined]
        truth_planets = list(task.config.planets)
        star_mass = float(task.config.star.M_star_sun)

        def _planet_summary(p):
            return {
                "P_days": float(p.P_days),
                "m_sin_i_mjup": float(p.m_sin_i_mjup),
                "e": float(p.e),
                "l_rad": float(p.l_rad),
            }

        from stargazer.config import PlanetParams
        from stargazer.matching import planet_score_components
        from stargazer.utils_units import semi_amplitude_ms

        submitted_planets = action.get("planets", [])
        guess_planets = [
            PlanetParams(
                P_days=float(p.get("P_days", 0.0)),
                m_sin_i_mjup=float(p.get("m_sin_i_mjup", 0.1)),
                e=float(p.get("e", 0.0)),
                inc_rad=float(p.get("inc_rad", 0.0)),
                Omega_rad=float(p.get("Omega_rad", 0.0)),
                omega_rad=float(p.get("omega_rad", 0.0)),
                l_rad=float(p.get("l_rad", 0.0)),
                m_true_mjup=None,
            )
            for p in submitted_planets
        ]

        truth_list = []
        for p in truth_planets:
            entry = _planet_summary(p)
            entry["K_ms"] = float(semi_amplitude_ms(p.m_sin_i_mjup, p.P_days, p.e, star_mass))
            truth_list.append(entry)

        guess_list = []
        for p in guess_planets:
            entry = _planet_summary(p)
            entry["K_ms"] = float(semi_amplitude_ms(p.m_sin_i_mjup, p.P_days, p.e, star_mass))
            guess_list.append(entry)

        import numpy as np
        times_arr = np.asarray(task.observations.times_days, dtype=float)
        assignment = (metrics.get("matching") or {}).get("assignment") or {}
        pairs = assignment.get("pairs") or []
        pair_details = []
        for t_idx, g_idx, dist in pairs:
            t_idx_i = int(t_idx)
            g_idx_i = int(g_idx)
            if 0 <= t_idx_i < len(truth_planets) and 0 <= g_idx_i < len(guess_planets):
                comps = planet_score_components(truth_planets[t_idx_i], guess_planets[g_idx_i], star_mass, times_days=times_arr)
            else:
                comps = None
            pair_details.append(
                {
                    "guess_idx": g_idx_i,
                    "distance": float(dist),
                    "components": comps,
                }
            )

        comparison = {
            "submitted_planets": guess_list,
            "matched_pairs": pair_details,
            "unmatched_guess": assignment.get("unmatched_guess"),
        }
    except Exception:
        comparison = None

    # Strip truth-revealing fields from matching and components before exposing to agent
    raw_matching = metrics.get("matching") or {}
    agent_matching = {k: v for k, v in raw_matching.items() if k != "unmatched_truth"}
    raw_components = dict(metrics.get("components") or {})
    raw_components.pop("count", None)
    raw_success_details = dict(info.get("success_details") or {})
    raw_success_details.pop("count_term", None)

    summary = {
        "reward": float(reward),
        "done": bool(done),
        "success": bool(info.get("success", False)),
        "success_details": raw_success_details,
        "semantic_check": semantic_check,
        "components": raw_components,
        "residuals": metrics.get("residuals"),
        "matching": agent_matching,
        "comparison": comparison,
    }

    result_str = json.dumps(summary, indent=2)

    return result_str, obs_next, reward, done, info
