#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import radvel
from astropy.timeseries import LombScargle
from scipy.optimize import least_squares

from stargazer.limits import DEFAULT_SUBMISSION_MAX_PLANETS


EPS = 1e-12
DEFAULT_BANK_SUBDIR = "stargazer/Stargazer_synthetic_task"


@dataclass
class PlanetFit:
    P_days: float
    K_ms: float
    e: float
    omega_rad: float
    l_rad: float


@dataclass
class ModelFit:
    planets: list[PlanetFit]
    gamma_per_inst: dict[str, float]
    rv_model_ms: np.ndarray
    residuals_ms: np.ndarray
    chi2: float
    loglike: float
    bic: float


def wrap_angle(x: float) -> float:
    return float(x % (2.0 * math.pi))


def inverse_semi_amplitude_mjup(k_ms: float, period_days: float, e: float, star_mass_sun: float) -> float:
    p_years = period_days / 365.25
    denom = 28.4329 * (p_years ** (-1.0 / 3.0)) * (star_mass_sun ** (-2.0 / 3.0))
    return float(max(k_ms, 1e-6) * math.sqrt(max(1e-8, 1.0 - e * e)) / max(denom, 1e-12))


def weighted_mean(values: np.ndarray, sigmas: np.ndarray) -> float:
    weights = 1.0 / np.maximum(sigmas, 1e-6) ** 2
    return float(np.sum(values * weights) / np.sum(weights))


def detrend_per_instrument(y_ms: np.ndarray, sigmas_ms: np.ndarray, instruments: np.ndarray) -> np.ndarray:
    centered = y_ms.astype(float).copy()
    for inst in np.unique(instruments):
        mask = instruments == inst
        centered[mask] -= weighted_mean(centered[mask], sigmas_ms[mask])
    return centered


def null_model_fit(y_ms: np.ndarray, sigmas_ms: np.ndarray, instruments: np.ndarray) -> ModelFit:
    gamma_per_inst: dict[str, float] = {}
    rv_model = np.zeros_like(y_ms, dtype=float)
    for inst in np.unique(instruments):
        mask = instruments == inst
        gamma = weighted_mean(y_ms[mask], sigmas_ms[mask])
        gamma_per_inst[str(inst)] = gamma
        rv_model[mask] = gamma
    residuals = y_ms - rv_model
    chi2 = float(np.sum((residuals / np.maximum(sigmas_ms, 1e-6)) ** 2))
    loglike = float(-0.5 * np.sum((residuals / np.maximum(sigmas_ms, 1e-6)) ** 2 + np.log(2.0 * math.pi * np.maximum(sigmas_ms, 1e-6) ** 2)))
    bic = float(-2.0 * loglike + len(gamma_per_inst) * math.log(len(y_ms)))
    return ModelFit(
        planets=[],
        gamma_per_inst=gamma_per_inst,
        rv_model_ms=rv_model,
        residuals_ms=residuals,
        chi2=chi2,
        loglike=loglike,
        bic=bic,
    )


def choose_period_bounds(times_days: np.ndarray, min_period_days: float, max_period_days: float | None) -> tuple[float, float]:
    span = max(float(np.max(times_days) - np.min(times_days)), min_period_days * 3.0)
    max_period = max_period_days if max_period_days is not None else min(max(300.0, span * 1.5), 5000.0)
    max_period = max(float(max_period), min_period_days * 1.5)
    return float(min_period_days), float(max_period)


def top_period_candidates(
    times_days: np.ndarray,
    residuals_ms: np.ndarray,
    sigmas_ms: np.ndarray,
    min_period_days: float,
    max_period_days: float,
    top_k: int,
    min_log_separation: float,
    samples_per_peak: int,
) -> list[float]:
    ls = LombScargle(times_days, residuals_ms, dy=sigmas_ms, fit_mean=True, center_data=True)
    freq, power = ls.autopower(
        minimum_frequency=1.0 / max_period_days,
        maximum_frequency=1.0 / min_period_days,
        samples_per_peak=samples_per_peak,
    )
    order = np.argsort(power)[::-1]
    periods: list[float] = []
    for idx in order.tolist():
        period = float(1.0 / freq[idx])
        if not np.isfinite(period):
            continue
        if period < min_period_days or period > max_period_days:
            continue
        logp = math.log(period)
        if any(abs(logp - math.log(p)) < min_log_separation for p in periods):
            continue
        periods.append(period)
        if len(periods) >= top_k:
            break
    return periods


def weighted_circular_init(
    times_days: np.ndarray,
    residuals_ms: np.ndarray,
    sigmas_ms: np.ndarray,
    period_days: float,
) -> tuple[float, float]:
    t_ref = float(times_days[0])
    phase = 2.0 * math.pi * (times_days - t_ref) / period_days
    design = np.column_stack([np.cos(phase), np.sin(phase), np.ones_like(phase)])
    weights = 1.0 / np.maximum(sigmas_ms, 1e-6)
    aw = design * weights[:, None]
    bw = residuals_ms * weights
    coeffs, _, _, _ = np.linalg.lstsq(aw, bw, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    amplitude = max(math.hypot(a, b), 0.1)
    lambda0 = wrap_angle(math.atan2(-b, a))
    return amplitude, lambda0


def unpack_planet_params(
    params: np.ndarray,
    min_period_days: float,
    max_period_days: float,
) -> list[PlanetFit]:
    planets: list[PlanetFit] = []
    emax = math.sqrt(0.95)
    for i in range(len(params) // 5):
        log_p, log_k, h, k, lambda0 = [float(x) for x in params[i * 5 : (i + 1) * 5]]
        period = float(np.clip(np.exp(log_p), min_period_days, max_period_days))
        semi_amp = float(np.clip(np.exp(log_k), 0.05, 1e4))
        hk_norm = math.hypot(h, k)
        if hk_norm > emax:
            scale = emax / hk_norm
            h *= scale
            k *= scale
        e = float(np.clip(h * h + k * k, 0.0, 0.95))
        omega = wrap_angle(math.atan2(k, h)) if e > 1e-8 else 0.0
        planets.append(
            PlanetFit(
                P_days=period,
                K_ms=semi_amp,
                e=e,
                omega_rad=omega,
                l_rad=wrap_angle(lambda0),
            )
        )
    return planets


def model_from_planets(
    times_days: np.ndarray,
    y_ms: np.ndarray,
    sigmas_ms: np.ndarray,
    instruments: np.ndarray,
    planets: list[PlanetFit],
) -> ModelFit:
    t_ref = float(times_days[0])
    base_rv = np.zeros_like(y_ms, dtype=float)
    for planet in planets:
        m0 = wrap_angle(planet.l_rad - planet.omega_rad)
        tp = t_ref - (m0 / (2.0 * math.pi)) * planet.P_days
        base_rv += radvel.kepler.rv_drive(
            times_days,
            [planet.P_days, tp, planet.e, planet.omega_rad, planet.K_ms],
            use_c_kepler_solver=False,
        )

    gamma_per_inst: dict[str, float] = {}
    rv_model = base_rv.copy()
    for inst in np.unique(instruments):
        mask = instruments == inst
        gamma = weighted_mean(y_ms[mask] - base_rv[mask], sigmas_ms[mask])
        gamma_per_inst[str(inst)] = gamma
        rv_model[mask] += gamma

    residuals = y_ms - rv_model
    scaled = residuals / np.maximum(sigmas_ms, 1e-6)
    chi2 = float(np.sum(scaled**2))
    loglike = float(-0.5 * np.sum(scaled**2 + np.log(2.0 * math.pi * np.maximum(sigmas_ms, 1e-6) ** 2)))
    k_params = 5 * len(planets) + len(np.unique(instruments))
    bic = float(-2.0 * loglike + k_params * math.log(len(y_ms)))
    return ModelFit(
        planets=planets,
        gamma_per_inst=gamma_per_inst,
        rv_model_ms=rv_model,
        residuals_ms=residuals,
        chi2=chi2,
        loglike=loglike,
        bic=bic,
    )


def fit_planet_set(
    times_days: np.ndarray,
    y_ms: np.ndarray,
    sigmas_ms: np.ndarray,
    instruments: np.ndarray,
    init_params: np.ndarray,
    min_period_days: float,
    max_period_days: float,
    n_restarts: int,
    rng_seed: int,
) -> ModelFit:
    lower = []
    upper = []
    for _ in range(len(init_params) // 5):
        lower.extend([math.log(min_period_days), math.log(0.05), -1.0, -1.0, 0.0])
        upper.extend([math.log(max_period_days), math.log(1e4), 1.0, 1.0, 2.0 * math.pi])
    lower_arr = np.array(lower, dtype=float)
    upper_arr = np.array(upper, dtype=float)
    init_params = np.clip(init_params.astype(float), lower_arr + 1e-6, upper_arr - 1e-6)

    def residual_fn(theta: np.ndarray) -> np.ndarray:
        planets = unpack_planet_params(theta, min_period_days=min_period_days, max_period_days=max_period_days)
        fit = model_from_planets(times_days, y_ms, sigmas_ms, instruments, planets)
        return fit.residuals_ms / np.maximum(sigmas_ms, 1e-6)

    rng = np.random.default_rng(rng_seed)
    best_fit: ModelFit | None = None
    start_points = [init_params]
    for _ in range(max(0, n_restarts - 1)):
        trial = init_params.copy()
        for idx in range(len(trial) // 5):
            base = idx * 5
            trial[base + 0] += rng.normal(0.0, 0.08)
            trial[base + 1] += rng.normal(0.0, 0.15)
            trial[base + 2] += rng.normal(0.0, 0.08)
            trial[base + 3] += rng.normal(0.0, 0.08)
            trial[base + 4] = wrap_angle(trial[base + 4] + rng.normal(0.0, 0.35))
        start_points.append(np.clip(trial, lower_arr + 1e-6, upper_arr - 1e-6))

    for start in start_points:
        result = least_squares(
            residual_fn,
            x0=start,
            bounds=(lower_arr, upper_arr),
            method="trf",
            max_nfev=3000,
            x_scale="jac",
        )
        planets = unpack_planet_params(result.x, min_period_days=min_period_days, max_period_days=max_period_days)
        fit = model_from_planets(times_days, y_ms, sigmas_ms, instruments, planets)
        if best_fit is None or fit.bic < best_fit.bic:
            best_fit = fit

    if best_fit is None:
        raise RuntimeError("least_squares produced no fit")
    return best_fit


def greedy_fit_task(
    times_days: np.ndarray,
    y_ms: np.ndarray,
    sigmas_ms: np.ndarray,
    instruments: np.ndarray,
    max_planets: int,
    peak_branch: int,
    restarts_per_candidate: int,
    min_period_days: float,
    max_period_days: float,
    min_bic_improvement: float,
    rng_seed: int,
) -> ModelFit:
    current = null_model_fit(y_ms, sigmas_ms, instruments)
    n_obs = len(y_ms)
    if min_bic_improvement <= 0.0:
        min_bic_improvement = 6.0 * math.log(n_obs)
    for depth in range(max_planets):
        candidate_periods = top_period_candidates(
            times_days=times_days,
            residuals_ms=current.residuals_ms,
            sigmas_ms=sigmas_ms,
            min_period_days=min_period_days,
            max_period_days=max_period_days,
            top_k=peak_branch,
            min_log_separation=0.08,
            samples_per_peak=8,
        )
        if not candidate_periods:
            break

        best_next: ModelFit | None = None
        for branch_idx, period in enumerate(candidate_periods):
            amp, lambda0 = weighted_circular_init(times_days, current.residuals_ms, sigmas_ms, period)
            init = []
            for planet in current.planets:
                h = math.sqrt(max(planet.e, 0.0)) * math.cos(planet.omega_rad)
                k = math.sqrt(max(planet.e, 0.0)) * math.sin(planet.omega_rad)
                init.extend([math.log(planet.P_days), math.log(max(planet.K_ms, 0.05)), h, k, planet.l_rad])
            init.extend([math.log(period), math.log(max(amp, 0.05)), 0.0, 0.0, lambda0])
            fit = fit_planet_set(
                times_days=times_days,
                y_ms=y_ms,
                sigmas_ms=sigmas_ms,
                instruments=instruments,
                init_params=np.array(init, dtype=float),
                min_period_days=min_period_days,
                max_period_days=max_period_days,
                n_restarts=restarts_per_candidate,
                rng_seed=rng_seed + depth * 100 + branch_idx,
            )
            if best_next is None or fit.bic < best_next.bic:
                best_next = fit

        if best_next is None:
            break
        if current.bic - best_next.bic < min_bic_improvement:
            break
        current = best_next
    return current


def submission_from_fit(planets: list[PlanetFit], star_mass_sun: float) -> dict[str, Any]:
    serialised = []
    for planet in planets:
        serialised.append(
            {
                "P_days": round(float(planet.P_days), 8),
                "m_sin_i_mjup": round(inverse_semi_amplitude_mjup(planet.K_ms, planet.P_days, planet.e, star_mass_sun), 8),
                "e": round(float(planet.e), 8),
                "omega_rad": round(float(planet.omega_rad), 8),
                "Omega_rad": 0.0,
                "l_rad": round(float(planet.l_rad), 8),
            }
        )
    return {"planets": serialised}


def difficulty_tier(difficulty: int) -> str:
    if difficulty <= 2:
        return "Easy"
    if difficulty <= 6:
        return "Medium"
    return "Hard"


def task_ids_from_args(bank: Any, task_id_args: list[str], limit: int | None) -> list[str]:
    if task_id_args:
        items: list[str] = []
        for raw in task_id_args:
            for token in raw.split(","):
                token = token.strip()
                if token:
                    items.append(token)
        return items
    task_ids = bank.list_tasks()
    if limit is not None:
        task_ids = task_ids[:limit]
    return task_ids


def import_stargazer(stargazer_root: str) -> tuple[Any, Any]:
    sys.path.insert(0, stargazer_root)
    try:
        from stargazer.bank import TaskBank
        from stargazer.env import RvEnv
        return TaskBank, RvEnv
    finally:
        if sys.path and sys.path[0] == stargazer_root:
            sys.path.pop(0)


def run_one_task(
    stargazer_root: str,
    bank_dir: str,
    task_id: str,
    max_planets: int,
    peak_branch: int,
    restarts_per_candidate: int,
    min_period_days: float,
    max_period_days: float | None,
    min_bic_improvement: float,
    seed: int,
    output_dir: str,
) -> dict[str, Any]:
    TaskBank, RvEnv = import_stargazer(stargazer_root)
    bank = TaskBank(bank_dir)
    task = bank.load_task(task_id)

    times_days = np.array(task.observations.times_days, dtype=float)
    y_ms = np.array(task.observations.rvs_ms, dtype=float)
    sigmas_ms = np.array(task.observations.sigmas_ms, dtype=float)
    instruments = np.array(task.observations.instruments)

    min_period, max_period = choose_period_bounds(times_days, min_period_days=min_period_days, max_period_days=max_period_days)
    fit = greedy_fit_task(
        times_days=times_days,
        y_ms=y_ms,
        sigmas_ms=sigmas_ms,
        instruments=instruments,
        max_planets=max_planets,
        peak_branch=peak_branch,
        restarts_per_candidate=restarts_per_candidate,
        min_period_days=min_period,
        max_period_days=max_period,
        min_bic_improvement=min_bic_improvement,
        rng_seed=seed,
    )

    submission = submission_from_fit(fit.planets, star_mass_sun=float(task.config.star.M_star_sun))
    env = RvEnv(task=task, submission_mode="params_and_model", max_steps=1)
    env.reset()
    _, reward, _, info = env.step(submission)
    success_details = info.get("success_details", {}) or {}
    components = ((info.get("metrics") or {}).get("components") or {})

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    subdir = output_path / "submissions"
    subdir.mkdir(parents=True, exist_ok=True)
    with open(subdir / f"{task_id}.json", "w") as f:
        json.dump(submission, f, indent=2)

    return {
        "task_id": task_id,
        "difficulty": int(task.truth_difficulty),
        "tier": difficulty_tier(int(task.truth_difficulty)),
        "is_real": bool(task_id.startswith("real_")),
        "n_truth": len(task.config.planets),
        "n_pred": len(fit.planets),
        "success": bool(info.get("success", False)),
        "reward": float(reward),
        "match_score": success_details.get("match_score"),
        "ok_match": success_details.get("ok_match"),
        "ok_count": success_details.get("ok_count"),
        "ok_rms": success_details.get("ok_rms"),
        "ok_delta_bic": success_details.get("ok_delta_bic"),
        "rms_ms": success_details.get("rms_ms"),
        "max_rms_ms": success_details.get("max_rms_ms"),
        "delta_bic_per_point": success_details.get("delta_bic_per_point"),
        "bic": float(fit.bic),
        "chi2": float(fit.chi2),
        "pred_periods_days": [round(p.P_days, 6) for p in fit.planets],
        "submission_path": str(subdir / f"{task_id}.json"),
        "metrics_components": components,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"n_tasks": 0, "pass_rate": None, "avg_pred_planets": None}
        return {
            "n_tasks": len(rows),
            "pass_rate": float(np.mean([float(row["success"]) for row in rows])),
            "avg_pred_planets": float(np.mean([row["n_pred"] for row in rows])),
            "avg_match_score": float(np.mean([row["match_score"] for row in rows if row["match_score"] is not None])) if any(row["match_score"] is not None for row in rows) else None,
            "avg_delta_bic_per_point": float(np.mean([row["delta_bic_per_point"] for row in rows if row["delta_bic_per_point"] is not None])) if any(row["delta_bic_per_point"] is not None for row in rows) else None,
            "avg_rms_ms": float(np.mean([row["rms_ms"] for row in rows if row["rms_ms"] is not None])) if any(row["rms_ms"] is not None for row in rows) else None,
        }

    by_tier = {}
    for tier in ("Easy", "Medium", "Hard"):
        by_tier[tier] = aggregate([row for row in results if row["tier"] == tier])

    real_rows = [row for row in results if row["is_real"]]
    synth_rows = [row for row in results if not row["is_real"]]
    return {
        "overall": aggregate(results),
        "by_tier": by_tier,
        "synthetic": aggregate(synth_rows),
        "real": aggregate(real_rows),
    }


def write_outputs(output_dir: Path, results: list[dict[str, Any]], summary: dict[str, Any], args_dict: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump({"args": args_dict, "summary": summary, "results": results}, f, indent=2)

    csv_rows = []
    for row in results:
        csv_rows.append(
            {
                "task_id": row["task_id"],
                "tier": row["tier"],
                "difficulty": row["difficulty"],
                "is_real": row["is_real"],
                "n_truth": row["n_truth"],
                "n_pred": row["n_pred"],
                "success": row["success"],
                "reward": row["reward"],
                "match_score": row["match_score"],
                "ok_match": row["ok_match"],
                "ok_count": row["ok_count"],
                "ok_rms": row["ok_rms"],
                "ok_delta_bic": row["ok_delta_bic"],
                "rms_ms": row["rms_ms"],
                "max_rms_ms": row["max_rms_ms"],
                "delta_bic_per_point": row["delta_bic_per_point"],
                "pred_periods_days": ";".join(str(x) for x in row["pred_periods_days"]),
                "submission_path": row["submission_path"],
            }
        )

    with open(output_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        if csv_rows:
            writer.writeheader()
            writer.writerows(csv_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classical Lomb-Scargle + greedy multi-Keplerian baseline for Stargazer.")
    parser.add_argument("--stargazer-root", required=True, help="Path to the stargazer repository root.")
    parser.add_argument("--bank-dir", default=None, help="Override task bank directory. Defaults to <stargazer-root>/stargazer/Stargazer_synthetic_task.")
    parser.add_argument("--task-id", action="append", default=[], help="Task id to run. Repeat or pass a comma-separated list.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N tasks from the bank when --task-id is omitted.")
    parser.add_argument("--output-dir", default="classical_baseline/out", help="Directory for JSON, CSV, and per-task submissions.")
    parser.add_argument("--workers", type=int, default=1, help="Process-level parallelism.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for restart jitter.")
    parser.add_argument(
        "--max-planets",
        type=int,
        default=DEFAULT_SUBMISSION_MAX_PLANETS,
        help="Maximum planets to fit greedily. Default matches the 7-planet real-task upper bound.",
    )
    parser.add_argument("--peak-branch", type=int, default=4, help="Top periodogram peaks evaluated at each depth.")
    parser.add_argument("--restarts-per-candidate", type=int, default=3, help="Least-squares restarts per candidate model.")
    parser.add_argument("--min-period-days", type=float, default=2.0, help="Minimum searched orbital period.")
    parser.add_argument("--max-period-days", type=float, default=None, help="Maximum searched orbital period. Default adapts to task span.")
    parser.add_argument("--min-bic-improvement", type=float, default=0.0, help="Minimum total BIC improvement to add a planet. When 0 (default), uses 6 * ln(N) which scales with sample size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stargazer_root = str(Path(args.stargazer_root).expanduser().resolve())
    bank_dir = str(Path(args.bank_dir).expanduser().resolve()) if args.bank_dir else str(Path(stargazer_root) / DEFAULT_BANK_SUBDIR)
    output_dir = Path(args.output_dir).expanduser().resolve()

    TaskBank, _ = import_stargazer(stargazer_root)
    bank = TaskBank(bank_dir)
    task_ids = task_ids_from_args(bank, args.task_id, args.limit)
    if not task_ids:
        raise SystemExit("No tasks selected.")

    run_kwargs = {
        "stargazer_root": stargazer_root,
        "bank_dir": bank_dir,
        "max_planets": args.max_planets,
        "peak_branch": args.peak_branch,
        "restarts_per_candidate": args.restarts_per_candidate,
        "min_period_days": args.min_period_days,
        "max_period_days": args.max_period_days,
        "min_bic_improvement": args.min_bic_improvement,
        "output_dir": str(output_dir),
    }

    results: list[dict[str, Any]] = []
    if args.workers <= 1:
        for idx, task_id in enumerate(task_ids):
            result = run_one_task(task_id=task_id, seed=args.seed + idx, **run_kwargs)
            results.append(result)
            print(f"[{idx + 1}/{len(task_ids)}] {task_id}: success={int(result['success'])} n_pred={result['n_pred']} match={result['match_score']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_one_task, task_id=task_id, seed=args.seed + idx, **run_kwargs): task_id
                for idx, task_id in enumerate(task_ids)
            }
            for count, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                print(f"[{count}/{len(task_ids)}] {result['task_id']}: success={int(result['success'])} n_pred={result['n_pred']} match={result['match_score']}")
        results.sort(key=lambda row: row["task_id"])

    summary = summarize(results)
    write_outputs(output_dir, results, summary, args_dict=vars(args))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
