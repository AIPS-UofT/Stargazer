import os
import tempfile
from stargazer.bank import TaskBank
from stargazer.env import RvEnv
from stargazer.benchmarks.baselines import baseline_null_model, baseline_one_sine
from stargazer.agents.common import canonicalize_plan, plan_to_action, validate_submission_semantics
from stargazer.config import GPParams, InstrumentParams, NoiseParams, Observations, ObservingSchedule, PlanetParams, StarParams, SystemConfig, Task
from stargazer.engine_rebound import simulate_clean_rv
from stargazer.evaluator import evaluate_submission
from stargazer.forward_keplerian import simulate_rv_keplerian
import numpy as np

def test_env_step_with_bank():
    root = os.path.join(os.path.dirname(__file__), "..", "stargazer_bank")
    bank = TaskBank(root)
    # Ensure at least one sample exists
    ids = bank.list_tasks()
    assert isinstance(ids, list)
    if ids:
        task = bank.load_task(ids[0])
        env = RvEnv(task=task, submission_mode="model_only", max_steps=1)
        obs, info = env.reset()
        act = baseline_null_model(obs)
        obs2, reward, done, info2 = env.step(act)
        assert isinstance(reward, float)
        assert done is True


def test_baseline_one_sine_returns_period_and_rms_diagnostics():
    t = np.linspace(0.0, 40.0, 200)
    p_true = 5.0
    y = 2.0 + 12.0 * np.sin(2.0 * np.pi * t / p_true + 0.3)
    s = np.ones_like(t)
    out = baseline_one_sine(
        {
            "times_days": t.tolist(),
            "rvs_ms": y.tolist(),
            "sigmas_ms": s.tolist(),
        }
    )
    assert "period_days" in out
    assert "rms_ms" in out
    assert "semi_amplitude_ms" in out
    assert "phase_rad" in out
    assert "rv_offset_ms" in out
    assert abs(out["period_days"] - p_true) < 0.25
    assert out["rms_ms"] < 1.0
    assert len(out["rv_model"]) == len(t)


def _make_simple_task(task_id: str, hints: dict | None = None) -> Task:
    times = [0.0, 1.0, 2.0, 3.0]
    cfg = SystemConfig(
        star=StarParams(M_star_sun=1.0, gamma_ms=0.0),
        planets=[],
        schedule=ObservingSchedule(times_days=times, instruments=["instA"] * len(times)),
        instruments=[InstrumentParams(label="instA", gamma_ms=0.0, sigma_white_ms=2.0, sigma_jitter_ms=0.0)],
        noise=NoiseParams(sigma_white_ms=2.0, sigma_jitter_ms=0.0, gp=GPParams(use_gp=False)),
        engine="keplerian",
        los_axis="x",
        integrator_preference="whfast",
    )
    obs = Observations(
        times_days=times,
        rvs_ms=[0.0, 0.0, 0.0, 0.0],
        sigmas_ms=[2.0, 2.0, 2.0, 2.0],
        instruments=["instA"] * len(times),
    )
    return Task(
        task_id=task_id,
        config=cfg,
        observations=obs,
        truth_difficulty=1,
        difficulty_details={},
        meta={"hints": hints or {}},
    )


def test_env_success_threshold_uses_task_max_rms_hint_when_available():
    task = _make_simple_task("hinted_rms", hints={"max_rms_ms": 10.0})
    env = RvEnv(task=task, submission_mode="model_only", max_steps=1)
    obs, _ = env.reset()
    _, _, _, info = env.step(baseline_null_model(obs))
    assert np.isclose(info["success_details"]["max_rms_ms"], 10.0)


def test_env_match_threshold_uses_task_target_match_hint_when_available():
    task = _make_simple_task("hinted_match", hints={"target_match_score": 0.95})
    env = RvEnv(task=task, submission_mode="params_only", max_steps=1)
    env.reset()
    _, _, _, info = env.step({"planets": []})
    assert np.isclose(info["success_details"]["min_match_score"], 0.95)


def test_plan_to_action_accepts_stargazer_planet_format():
    obs = {
        "times_days": [0.0, 1.0, 2.0],
        "rvs_ms": [0.0, 0.0, 0.0],
        "sigmas_ms": [1.0, 1.0, 1.0],
    }
    plan = {
        "planets": [
            {
                "P_days": 10.0,
                "m_sin_i_mjup": 1.0,
                "e": 0.1,
                "omega_rad": 0.2,
                "l_rad": 1.1,
            }
        ]
    }
    action = plan_to_action(plan, obs, max_planets=3, submission_mode="params_only")
    assert "planets" in action
    assert action["planets"][0]["P_days"] == 10.0


def test_plan_to_action_preserves_submitted_geometry_fields():
    obs = {
        "times_days": [0.0, 1.0, 2.0],
        "rvs_ms": [0.0, 0.0, 0.0],
        "sigmas_ms": [1.0, 1.0, 1.0],
    }
    plan = {
        "planets": [
            {
                "P_days": 10.0,
                "m_sin_i_mjup": 1.0,
                "e": 0.1,
                "inc_rad": 1.234,
                "Omega_rad": 2.345,
                "omega_rad": 0.2,
                "l_rad": 1.1,
            }
        ]
    }
    action = plan_to_action(plan, obs, max_planets=3, submission_mode="params_only")
    p = action["planets"][0]
    assert np.isclose(p["inc_rad"], 1.234)
    assert np.isclose(p["Omega_rad"], 2.345)


def test_plan_to_action_treats_null_numeric_fields_as_missing():
    obs = {
        "times_days": [0.0, 1.0, 2.0],
        "rvs_ms": [1.0, 2.0, 3.0],
        "sigmas_ms": [1.0, 1.0, 1.0],
    }
    plan = {
        "rv_offset_ms": None,
        "noise_jitter_ms": None,
        "planets": [{"period_days": 10.0, "semi_amplitude_ms": 0.0, "phase_rad": 0.0}],
    }
    action = plan_to_action(plan, obs, max_planets=3, submission_mode="params_and_model")
    assert isinstance(action["rv_model"], list)
    assert len(action["rv_model"]) == len(obs["rvs_ms"])


def test_plan_to_action_uses_rebound_forward_model_when_meta_available():
    times = [0.0, 2.5, 5.0, 7.5, 10.0]
    obs = {
        "times_days": times,
        "rvs_ms": [0.0] * len(times),
        "sigmas_ms": [1.0] * len(times),
        "meta": {"star_mass_sun": 1.0, "los_axis": "x", "integrator_preference": "whfast", "engine": "rebound"},
    }
    plan = {
        "rv_offset_ms": 0.0,
        "planets": [
            {
                "P_days": 10.0,
                "m_sin_i_mjup": 1.0,
                "e": 0.0,
                "inc_rad": float(np.pi / 2.0),
                "Omega_rad": 0.0,
                "omega_rad": 0.0,
                "l_rad": 0.0,
            }
        ],
    }
    action = plan_to_action(plan, obs, max_planets=3, submission_mode="model_only")
    # plan_to_action applies an Omega correction so that REBOUND's LOS axis
    # matches the Keplerian RV convention.  Verify against the Keplerian
    # forward model (the canonical scoring reference), allowing a small
    # tolerance for N-body vs analytic differences.
    from stargazer.forward_keplerian import simulate_rv_keplerian
    pp = PlanetParams(**plan["planets"][0])
    expected_kep = simulate_rv_keplerian(
        [pp], np.asarray(times, dtype=float), 1.0, gamma_ms=0.0
    )
    assert np.allclose(action["rv_model"], expected_kep, atol=0.5)


def test_submission_semantic_validator_allows_l_rad_with_conflicting_legacy_phase():
    obs = {
        "times_days": [100.0, 101.0, 102.0],
        "rvs_ms": [0.0, 0.0, 0.0],
        "sigmas_ms": [1.0, 1.0, 1.0],
    }
    plan = {
        "planets": [
            {
                "P_days": 10.0,
                "m_sin_i_mjup": 1.0,
                "e": 0.1,
                "omega_rad": 0.0,
                "l_rad": 0.0,
                "phase_rad": 2.4,  # Inconsistent with l_rad
            }
        ]
    }
    report = validate_submission_semantics(plan, obs)
    assert report["ok"] is True
    assert any("treated as canonical" in w for w in report["warnings"])


def test_plan_to_action_uses_l_rad_as_canonical_when_legacy_conflicts():
    obs = {
        "times_days": [100.0, 101.0, 102.0],
        "rvs_ms": [0.0, 0.0, 0.0],
        "sigmas_ms": [1.0, 1.0, 1.0],
    }
    plan = {
        "planets": [
            {
                "P_days": 10.0,
                "m_sin_i_mjup": 1.0,
                "e": 0.1,
                "omega_rad": 0.5,
                "l_rad": 1.234,
                "phase_rad": 2.4,  # conflicting legacy field
                "phase_frac": 0.25,  # conflicting legacy field
            }
        ]
    }
    action = plan_to_action(plan, obs, max_planets=3, submission_mode="params_only")
    assert np.isclose(action["planets"][0]["l_rad"], 1.234)


def test_submission_semantic_validator_flags_phase_conflicts_without_l_rad():
    obs = {
        "times_days": [100.0, 101.0, 102.0],
        "rvs_ms": [0.0, 0.0, 0.0],
        "sigmas_ms": [1.0, 1.0, 1.0],
    }
    plan = {
        "planets": [
            {
                "P_days": 10.0,
                "m_sin_i_mjup": 1.0,
                "e": 0.1,
                "omega_rad": 0.0,
                "phase_rad": 2.4,
                "phase_deg": 0.0,
            }
        ]
    }
    report = validate_submission_semantics(plan, obs)
    assert report["ok"] is False
    assert any("phase semantics conflict" in e for e in report["errors"])


def test_submission_semantic_validator_uses_reference_epoch():
    obs = {
        "times_days": [123.4, 124.4],
        "rvs_ms": [0.0, 0.0],
        "sigmas_ms": [1.0, 1.0],
    }
    plan = {
        "planets": [
            {
                "P_days": 10.0,
                "m_sin_i_mjup": 1.0,
                "omega_rad": 0.5,
                "phase_rad": 1.0,
            }
        ]
    }
    report = validate_submission_semantics(plan, obs)
    assert report["ok"] is True
    assert report["t_ref_days"] == 123.4


def test_params_and_model_scoring_rebuilds_model_from_planets_not_submission_rv_model():
    times = np.linspace(0.0, 30.0, 64)
    truth_planet = PlanetParams(
        P_days=12.0,
        m_sin_i_mjup=0.7,
        e=0.2,
        inc_rad=float(np.pi / 2.0),
        Omega_rad=1.2,
        omega_rad=0.4,
        l_rad=2.1,
        m_true_mjup=None,
    )
    cfg = SystemConfig(
        star=StarParams(M_star_sun=1.0, gamma_ms=0.0),
        planets=[truth_planet],
        schedule=ObservingSchedule(times_days=times.tolist(), instruments=["instA"] * len(times)),
        instruments=[InstrumentParams(label="instA", gamma_ms=0.0, sigma_white_ms=1.0, sigma_jitter_ms=0.0)],
        noise=NoiseParams(sigma_white_ms=1.0, sigma_jitter_ms=0.0, gp=GPParams(use_gp=False)),
    )
    y = simulate_rv_keplerian([truth_planet], times, M_star_sun=1.0, gamma_ms=0.0)
    obs = Observations(
        times_days=times.tolist(),
        rvs_ms=y.tolist(),
        sigmas_ms=np.ones_like(times).tolist(),
        instruments=["instA"] * len(times),
    )
    submission = {
        "planets": [
            {
                "P_days": truth_planet.P_days,
                "m_sin_i_mjup": truth_planet.m_sin_i_mjup,
                "e": truth_planet.e,
                "Omega_rad": truth_planet.Omega_rad,
                "omega_rad": truth_planet.omega_rad,
                "l_rad": truth_planet.l_rad,
            }
        ],
        # Intentionally wrong; scorer should ignore this in params_and_model.
        "rv_model": np.zeros_like(times).tolist(),
        "noise": {"sigma_jitter_ms": 0.0},
    }
    reward, info = evaluate_submission(
        config=cfg,
        obs=obs,
        submission=submission,
        truth_planets=[truth_planet],
        reward_weights={"likelihood": 1.0, "delta_bic": 0.3, "neg_rms": 0.1, "match": 1.0, "count": 0.2},
        mode="params_and_model",
    )
    assert np.isfinite(reward)
    assert info["rv_model_source"] == "rv_only_keplerian_from_planets"
    assert info["residuals"]["rms"] < 1e-6


def test_lrad_and_phase_rad_plans_produce_same_score():
    times = np.linspace(0.0, 25.0, 50)
    obs = {
        "times_days": times.tolist(),
        "rvs_ms": np.zeros_like(times).tolist(),
        "sigmas_ms": np.ones_like(times).tolist(),
    }
    plan_l = {
        "planets": [
            {
                "P_days": 9.5,
                "m_sin_i_mjup": 0.5,
                "e": 0.1,
                "omega_rad": 0.8,
                "Omega_rad": 2.0,
                "l_rad": 1.7,
            }
        ]
    }
    m0 = (
        plan_l["planets"][0]["l_rad"]
        - (plan_l["planets"][0]["Omega_rad"] + plan_l["planets"][0]["omega_rad"])
    ) % (2.0 * np.pi)
    plan_phase = {
        "planets": [
            {
                "P_days": 9.5,
                "m_sin_i_mjup": 0.5,
                "e": 0.1,
                "omega_rad": 0.8,
                "Omega_rad": 2.0,
                "phase_rad": m0,
            }
        ]
    }
    action_l = plan_to_action(plan_l, obs, max_planets=3, submission_mode="params_and_model")
    action_phase = plan_to_action(plan_phase, obs, max_planets=3, submission_mode="params_and_model")
    assert np.allclose(action_l["rv_model"], action_phase["rv_model"], atol=1e-9)


def test_canonicalize_plan_drops_conflicting_phase_aliases():
    plan = {
        "planets": [
            {
                "P_days": 10.0,
                "period_days": 10.0,
                "e": 0.1,
                "eccentricity": 0.1,
                "omega_rad": 0.5,
                "l_rad": 1.2,
                "phase_rad": 2.4,
                "phase_deg": 137.0,
                "phase_frac": 0.25,
                "T0_days": 123.0,
            }
        ]
    }
    out = canonicalize_plan(plan)
    p = out["planets"][0]
    assert "period_days" not in p
    assert "eccentricity" not in p
    assert "phase_rad" not in p
    assert "phase_deg" not in p
    assert "phase_frac" not in p
    assert "T0_days" not in p


def test_canonicalize_plan_prefers_native_mass_over_legacy_amplitude():
    plan = {
        "planets": [
            {
                "P_days": 10.0,
                "m_sin_i_mjup": 0.5,
                "semi_amplitude_ms": 0.0,
                "K_ms": 123.0,
                "e": 0.1,
                "l_rad": 0.7,
            }
        ]
    }
    out = canonicalize_plan(plan)
    p = out["planets"][0]
    assert "semi_amplitude_ms" not in p
    assert "K_ms" not in p


def test_taskbank_load_applies_rv_only_compat_mapping():
    times = [0.0, 2.0, 5.0, 8.0, 11.0]
    planet = PlanetParams(
        P_days=10.0,
        m_sin_i_mjup=1.0,
        e=0.2,
        inc_rad=float(np.pi / 2.0),
        Omega_rad=1.1,
        omega_rad=0.6,
        l_rad=2.2,
        m_true_mjup=None,
    )
    cfg = SystemConfig(
        star=StarParams(M_star_sun=1.0, gamma_ms=0.0),
        planets=[planet],
        schedule=ObservingSchedule(times_days=times, instruments=["instA"] * len(times)),
        instruments=[InstrumentParams(label="instA", gamma_ms=0.0, sigma_white_ms=1.0, sigma_jitter_ms=0.0)],
        noise=NoiseParams(sigma_white_ms=1.0, sigma_jitter_ms=0.0, gp=GPParams(use_gp=False)),
        engine="rebound",
        los_axis="x",
        integrator_preference="whfast",
    )
    y_clean = simulate_clean_rv(cfg, times).tolist()
    task = Task(
        task_id="compat_case",
        config=cfg,
        observations=Observations(
            times_days=times,
            rvs_ms=y_clean,
            sigmas_ms=[1.0] * len(times),
            instruments=["instA"] * len(times),
        ),
        truth_difficulty=1,
        difficulty_details={},
        meta={},
    )

    with tempfile.TemporaryDirectory() as d:
        bank = TaskBank(d)
        bank.add_task(task)
        loaded = bank.load_task("compat_case")
        assert loaded.meta.get("rv_only_compat_applied") is True
        # After compat mapping, planets are converted to RV-only semantics:
        # Omega_rad -> 0, l_rad -> (l_rad - Omega_rad) % 2pi
        expected_l_rad_rv = (planet.l_rad - planet.Omega_rad) % (2.0 * np.pi)
        assert np.isclose(loaded.config.planets[0].Omega_rad, 0.0, atol=1e-12)
        assert np.isclose(loaded.config.planets[0].l_rad, expected_l_rad_rv, atol=1e-12)
        # Observations should match the RV-only Keplerian model of the converted planets
        target = simulate_rv_keplerian(loaded.config.planets, np.asarray(times, dtype=float), M_star_sun=1.0, gamma_ms=0.0)
        assert np.allclose(np.asarray(loaded.observations.rvs_ms), target, atol=1e-6)
