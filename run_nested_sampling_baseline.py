#!/usr/bin/env python3
"""
Nested-sampling baseline for Stargazer.

Strategy per task:
  1. Lomb-Scargle periodogram → top-K period candidates
  2. For n_planets in {0, ..., max_planets}:
       - Run dynesty nested sampling with broad priors centred on LS peaks
       - Collect log-evidence (logZ) for model comparison
  3. Select n_planets by Bayes factor (logZ difference)
  4. Use posterior median as point estimate → build submission
  5. Score via RvEnv

Designed to be a drop-in comparison to run_classical_baseline.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import dynesty
from astropy.timeseries import LombScargle
from scipy.optimize import minimize

from stargazer.limits import DEFAULT_SUBMISSION_MAX_PLANETS

warnings.filterwarnings("ignore", category=RuntimeWarning)

EPS = 1e-12
TWO_PI = 2.0 * math.pi

# ---------------------------------------------------------------------------
# Keplerian RV model (pure numpy, no radvel dependency for the sampler)
# ---------------------------------------------------------------------------

def kepler_solve(M: np.ndarray, e: float, tol: float = 1e-10, maxiter: int = 30) -> np.ndarray:
    """Solve Kepler's equation M = E - e*sin(E) via Newton-Raphson."""
    E = M.copy()
    for _ in range(maxiter):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def rv_single_planet(times: np.ndarray, t_ref: float,
                     P: float, K: float, e: float, omega: float, l_rad: float) -> np.ndarray:
    """Compute RV contribution of one planet."""
    n = TWO_PI / P
    M0 = l_rad - omega  # mean anomaly at t_ref
    M = M0 + n * (times - t_ref)
    E = kepler_solve(M % TWO_PI, e)
    nu = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0),
                           np.sqrt(1.0 - e) * np.cos(E / 2.0))
    return K * (np.cos(nu + omega) + e * np.cos(omega))


def rv_model(times: np.ndarray, t_ref: float, planet_params: list[dict],
             gamma_per_inst: dict[str, float], instruments: np.ndarray) -> np.ndarray:
    """Full RV model: sum of Keplerians + per-instrument gamma."""
    rv = np.zeros_like(times, dtype=float)
    for p in planet_params:
        rv += rv_single_planet(times, t_ref, p["P"], p["K"], p["e"], p["omega"], p["l"])
    for inst, gamma in gamma_per_inst.items():
        rv[instruments == inst] += gamma
    return rv


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------

def log_likelihood(times: np.ndarray, rvs: np.ndarray, sigmas: np.ndarray,
                   instruments: np.ndarray, t_ref: float,
                   planet_params: list[dict], gamma_per_inst: dict[str, float],
                   jitter: float = 0.0) -> float:
    model = rv_model(times, t_ref, planet_params, gamma_per_inst, instruments)
    var = sigmas**2 + jitter**2
    resid = rvs - model
    return float(-0.5 * np.sum(resid**2 / var + np.log(TWO_PI * var)))


# ---------------------------------------------------------------------------
# Prior / transform helpers
# ---------------------------------------------------------------------------

def inverse_semi_amplitude_mjup(K: float, P: float, e: float, M_star: float) -> float:
    p_yr = P / 365.25
    denom = 28.4329 * (p_yr ** (-1.0 / 3.0)) * (M_star ** (-2.0 / 3.0))
    return max(K, 1e-6) * math.sqrt(max(1e-8, 1.0 - e * e)) / max(denom, EPS)


# ---------------------------------------------------------------------------
# Nested sampling for a given n_planets
# ---------------------------------------------------------------------------

def run_nested_sampling(
    times: np.ndarray,
    rvs: np.ndarray,
    sigmas: np.ndarray,
    instruments: np.ndarray,
    inst_list: list[str],
    t_ref: float,
    n_planets: int,
    period_candidates: list[float],
    period_half_widths: list[float],
    rv_range: float,
    nlive: int = 200,
    dlogz: float = 0.5,
    seed: int = 42,
) -> dict:
    """Run dynesty for a model with n_planets planets.

    Parameters per planet: logP, K, sqrt_e*cos(w), sqrt_e*sin(w), l_rad
    Global: gamma per instrument, log_jitter
    """
    n_inst = len(inst_list)
    ndim = 5 * n_planets + n_inst + 1  # +1 for log_jitter

    # Period priors centred on LS candidates
    # Use broad uniform in log-space around each candidate
    P_centres = []
    P_log_lo = []
    P_log_hi = []
    for i in range(n_planets):
        if i < len(period_candidates):
            Pc = period_candidates[i]
            hw = period_half_widths[i] if i < len(period_half_widths) else 0.5
        else:
            Pc = 10.0
            hw = 1.0
        P_centres.append(Pc)
        P_log_lo.append(math.log(max(Pc * (1.0 - hw), 0.5)))
        P_log_hi.append(math.log(Pc * (1.0 + hw)))

    K_max = max(rv_range * 2.0, 20.0)

    def prior_transform(u):
        """Map unit cube [0,1]^ndim → physical parameters."""
        theta = np.empty(ndim)
        for i in range(n_planets):
            base = i * 5
            # logP: uniform in [log_lo, log_hi]
            theta[base + 0] = P_log_lo[i] + u[base + 0] * (P_log_hi[i] - P_log_lo[i])
            # K: uniform [0.1, K_max]
            theta[base + 1] = 0.1 + u[base + 1] * (K_max - 0.1)
            # sqrt_e * cos(omega): uniform [-1, 1]
            theta[base + 2] = -1.0 + 2.0 * u[base + 2]
            # sqrt_e * sin(omega): uniform [-1, 1]
            theta[base + 3] = -1.0 + 2.0 * u[base + 3]
            # l_rad: uniform [0, 2pi]
            theta[base + 4] = u[base + 4] * TWO_PI
        # gamma per instrument
        gamma_lo = -rv_range * 3.0
        gamma_hi = rv_range * 3.0
        for j in range(n_inst):
            idx = 5 * n_planets + j
            theta[idx] = gamma_lo + u[idx] * (gamma_hi - gamma_lo)
        # log_jitter: uniform [log(0.01), log(max(rv_range, 10))]
        jit_idx = 5 * n_planets + n_inst
        log_jit_lo = math.log(0.01)
        log_jit_hi = math.log(max(rv_range, 10.0))
        theta[jit_idx] = log_jit_lo + u[jit_idx] * (log_jit_hi - log_jit_lo)
        return theta

    def loglike(theta):
        planets = []
        for i in range(n_planets):
            base = i * 5
            logP = theta[base + 0]
            K = theta[base + 1]
            sec = theta[base + 2]  # sqrt(e)*cos(w)
            ses = theta[base + 3]  # sqrt(e)*sin(w)
            l = theta[base + 4]

            e = min(sec**2 + ses**2, 0.95)
            if e < 1e-8:
                omega = 0.0
            else:
                omega = math.atan2(ses, sec) % TWO_PI

            planets.append({"P": math.exp(logP), "K": K, "e": e, "omega": omega, "l": l})

        gamma_dict = {}
        for j, inst in enumerate(inst_list):
            gamma_dict[inst] = theta[5 * n_planets + j]

        jitter = math.exp(theta[5 * n_planets + n_inst])

        return log_likelihood(times, rvs, sigmas, instruments, t_ref,
                              planets, gamma_dict, jitter)

    sampler = dynesty.NestedSampler(
        loglike, prior_transform, ndim,
        nlive=nlive,
        sample="rslice",
        bootstrap=0,
        rstate=np.random.Generator(np.random.PCG64(seed)),
    )
    sampler.run_nested(dlogz=dlogz, print_progress=False)
    results = sampler.results

    # Extract best-fit (posterior weighted median)
    weights = np.exp(results.logwt - results.logwt.max())
    weights /= weights.sum()
    samples = results.samples

    # Weighted median for each parameter
    best = np.empty(ndim)
    for d in range(ndim):
        sorted_idx = np.argsort(samples[:, d])
        cumw = np.cumsum(weights[sorted_idx])
        med_idx = sorted_idx[np.searchsorted(cumw, 0.5)]
        best[d] = samples[med_idx, d]

    # Parse best-fit
    planets_out = []
    for i in range(n_planets):
        base = i * 5
        logP = best[base + 0]
        K = best[base + 1]
        sec = best[base + 2]
        ses = best[base + 3]
        l = best[base + 4]
        e = min(sec**2 + ses**2, 0.95)
        omega = math.atan2(ses, sec) % TWO_PI if e > 1e-8 else 0.0
        planets_out.append({"P": math.exp(logP), "K": K, "e": e, "omega": omega, "l": l % TWO_PI})

    gamma_out = {}
    for j, inst in enumerate(inst_list):
        gamma_out[inst] = float(best[5 * n_planets + j])

    logZ = float(results.logz[-1])
    logZ_err = float(results.logzerr[-1])

    return {
        "n_planets": n_planets,
        "logZ": logZ,
        "logZ_err": logZ_err,
        "planets": planets_out,
        "gamma": gamma_out,
        "jitter": float(math.exp(best[5 * n_planets + n_inst])),
        "n_samples": len(samples),
    }


# ---------------------------------------------------------------------------
# Top-level: fit one task
# ---------------------------------------------------------------------------

def fit_task_nested(
    times: np.ndarray,
    rvs: np.ndarray,
    sigmas: np.ndarray,
    instruments: np.ndarray,
    max_planets: int = DEFAULT_SUBMISSION_MAX_PLANETS,
    nlive: int = 200,
    seed: int = 42,
    min_log_bayes: float = 5.0,
) -> dict:
    t_ref = float(times[0])
    inst_list = sorted(set(instruments.tolist()))
    rv_range = float(np.ptp(rvs))

    # Lomb-Scargle to find candidate periods
    centered = rvs.copy()
    for inst in inst_list:
        mask = instruments == inst
        w = 1.0 / np.maximum(sigmas[mask], 1e-6)**2
        centered[mask] -= np.sum(centered[mask] * w) / np.sum(w)

    span = float(np.ptp(times))
    ls = LombScargle(times, centered, dy=sigmas, fit_mean=True)
    freq, power = ls.autopower(
        minimum_frequency=1.0 / min(span * 1.5, 5000.0),
        maximum_frequency=1.0 / 1.0,
        samples_per_peak=10,
    )
    periods_all = 1.0 / freq
    order = np.argsort(power)[::-1]

    # Pick top distinct peaks
    top_periods = []
    for idx in order:
        p = float(periods_all[idx])
        if not np.isfinite(p) or p < 1.0:
            continue
        if any(abs(math.log(p) - math.log(tp)) < 0.1 for tp in top_periods):
            continue
        top_periods.append(p)
        if len(top_periods) >= max_planets * 3:
            break

    # Half-widths for period priors (broader for less certain peaks)
    half_widths = [0.3] * len(top_periods)  # ±30% around LS peak

    # Run 0-planet model
    results_by_n = {}
    res0 = run_nested_sampling(
        times, rvs, sigmas, instruments, inst_list, t_ref,
        n_planets=0, period_candidates=[], period_half_widths=[],
        rv_range=rv_range, nlive=max(nlive // 2, 50), seed=seed,
    )
    results_by_n[0] = res0
    print(f"    n=0: logZ={res0['logZ']:.1f}")

    # Incrementally add planets, compare Bayes factor
    best_n = 0
    for np_ in range(1, max_planets + 1):
        # Use top periods for this number of planets
        pcands = top_periods[:np_]
        hws = half_widths[:np_]

        # Pad if not enough LS peaks
        while len(pcands) < np_:
            pcands.append(10.0 * np_)
            hws.append(0.8)

        res = run_nested_sampling(
            times, rvs, sigmas, instruments, inst_list, t_ref,
            n_planets=np_, period_candidates=pcands, period_half_widths=hws,
            rv_range=rv_range, nlive=nlive, seed=seed + np_ * 1000,
        )
        results_by_n[np_] = res
        delta_logZ = res["logZ"] - results_by_n[best_n]["logZ"]
        print(f"    n={np_}: logZ={res['logZ']:.1f} (Δ={delta_logZ:.1f} vs n={best_n})")

        if delta_logZ > min_log_bayes:
            best_n = np_

    return results_by_n[best_n]


# ---------------------------------------------------------------------------
# Submission + scoring
# ---------------------------------------------------------------------------

def submission_from_nested(result: dict, star_mass: float) -> dict:
    planets = []
    for p in result["planets"]:
        m_sini = inverse_semi_amplitude_mjup(p["K"], p["P"], p["e"], star_mass)
        planets.append({
            "P_days": round(p["P"], 8),
            "m_sin_i_mjup": round(m_sini, 8),
            "e": round(p["e"], 8),
            "omega_rad": round(p["omega"], 8),
            "Omega_rad": 0.0,
            "l_rad": round(p["l"], 8),
        })
    return {"planets": planets}


def import_stargazer(root: str):
    sys.path.insert(0, root)
    try:
        from stargazer.bank import TaskBank
        from stargazer.env import RvEnv
        return TaskBank, RvEnv
    finally:
        if sys.path and sys.path[0] == root:
            sys.path.pop(0)


def difficulty_tier(d: int) -> str:
    if d <= 2: return "Easy"
    if d <= 6: return "Medium"
    return "Hard"


def run_one_task(
    stargazer_root: str,
    bank_dir: str,
    task_id: str,
    max_planets: int,
    nlive: int,
    min_log_bayes: float,
    seed: int,
    output_dir: str,
) -> dict[str, Any]:
    TaskBank, RvEnv = import_stargazer(stargazer_root)
    bank = TaskBank(bank_dir)
    task = bank.load_task(task_id)

    times = np.array(task.observations.times_days, dtype=float)
    rvs = np.array(task.observations.rvs_ms, dtype=float)
    sigmas = np.array(task.observations.sigmas_ms, dtype=float)
    instruments = np.array(task.observations.instruments)

    print(f"  [{task_id}] n_obs={len(times)} span={np.ptp(times):.0f}d")
    result = fit_task_nested(
        times, rvs, sigmas, instruments,
        max_planets=max_planets, nlive=nlive, seed=seed,
        min_log_bayes=min_log_bayes,
    )

    submission = submission_from_nested(result, float(task.config.star.M_star_sun))

    # Score
    env = RvEnv(task=task, submission_mode="params_and_model", max_steps=1)
    env.reset()
    _, reward, _, info = env.step(submission)
    sd = info.get("success_details", {}) or {}
    components = ((info.get("metrics") or {}).get("components") or {})

    # Save submission
    out = Path(output_dir)
    (out / "submissions").mkdir(parents=True, exist_ok=True)
    with open(out / "submissions" / f"{task_id}.json", "w") as f:
        json.dump(submission, f, indent=2)

    return {
        "task_id": task_id,
        "difficulty": int(task.truth_difficulty),
        "tier": difficulty_tier(int(task.truth_difficulty)),
        "is_real": task_id.startswith("real_"),
        "n_truth": len(task.config.planets),
        "n_pred": len(result["planets"]),
        "success": bool(info.get("success", False)),
        "reward": float(reward),
        "match_score": sd.get("match_score"),
        "ok_match": sd.get("ok_match"),
        "ok_count": sd.get("ok_count"),
        "ok_rms": sd.get("ok_rms"),
        "ok_delta_bic": sd.get("ok_delta_bic"),
        "rms_ms": sd.get("rms_ms"),
        "max_rms_ms": sd.get("max_rms_ms"),
        "delta_bic_per_point": sd.get("delta_bic_per_point"),
        "logZ": result["logZ"],
        "n_planets_selected": result["n_planets"],
        "pred_periods_days": [round(p["P"], 6) for p in result["planets"]],
        "metrics_components": components,
    }


def summarize(results: list[dict]) -> dict:
    def agg(rows):
        if not rows:
            return {"n_tasks": 0, "pass_rate": None}
        return {
            "n_tasks": len(rows),
            "pass_rate": float(np.mean([float(r["success"]) for r in rows])),
            "avg_match_score": float(np.mean([r["match_score"] for r in rows if r["match_score"] is not None])) if any(r["match_score"] is not None for r in rows) else None,
            "avg_rms_ms": float(np.mean([r["rms_ms"] for r in rows if r["rms_ms"] is not None])) if any(r["rms_ms"] is not None for r in rows) else None,
        }
    by_tier = {t: agg([r for r in results if r["tier"] == t]) for t in ("Easy", "Medium", "Hard")}
    return {"overall": agg(results), "by_tier": by_tier}


def main():
    p = argparse.ArgumentParser(description="Nested-sampling baseline for Stargazer")
    p.add_argument("--stargazer-root", required=True)
    p.add_argument("--bank-dir", default=None)
    p.add_argument("--task-id", action="append", default=[])
    p.add_argument("--diff", type=int, nargs="+", default=None, help="Filter by difficulty level(s)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-dir", default="nested_sampling_baseline/out")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-planets", type=int, default=DEFAULT_SUBMISSION_MAX_PLANETS)
    p.add_argument("--nlive", type=int, default=200)
    p.add_argument("--min-log-bayes", type=float, default=5.0, help="Min ln(Bayes factor) to add a planet")
    args = p.parse_args()

    root = str(Path(args.stargazer_root).resolve())
    bank_dir = str(Path(args.bank_dir).resolve()) if args.bank_dir else str(Path(root) / "stargazer/Stargazer_synthetic_task")
    output_dir = Path(args.output_dir).resolve()

    TaskBank, _ = import_stargazer(root)
    bank = TaskBank(bank_dir)
    task_ids = bank.list_tasks()

    # Filter by --task-id
    if args.task_id:
        explicit = []
        for raw in args.task_id:
            for tok in raw.split(","):
                tok = tok.strip()
                if tok:
                    explicit.append(tok)
        task_ids = explicit

    # Filter by --diff
    if args.diff is not None:
        filtered = []
        for tid in task_ids:
            t = bank.load_task(tid)
            if t.truth_difficulty in args.diff:
                filtered.append(tid)
        task_ids = filtered

    if args.limit:
        task_ids = task_ids[:args.limit]

    if not task_ids:
        raise SystemExit("No tasks selected.")

    print(f"Running nested-sampling baseline on {len(task_ids)} tasks")
    print(f"  nlive={args.nlive}, max_planets={args.max_planets}, min_log_bayes={args.min_log_bayes}")

    results = []
    if args.workers <= 1:
        for idx, tid in enumerate(task_ids):
            result = run_one_task(
                stargazer_root=root, bank_dir=bank_dir, task_id=tid,
                max_planets=args.max_planets, nlive=args.nlive,
                min_log_bayes=args.min_log_bayes, seed=args.seed + idx,
                output_dir=str(output_dir),
            )
            results.append(result)
            print(f"[{idx+1}/{len(task_ids)}] {tid}: success={int(result['success'])} "
                  f"n_pred={result['n_pred']} match={result['match_score']:.3f} "
                  f"reward={result['reward']:.2f}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_one_task,
                    stargazer_root=root, bank_dir=bank_dir, task_id=tid,
                    max_planets=args.max_planets, nlive=args.nlive,
                    min_log_bayes=args.min_log_bayes, seed=args.seed + idx,
                    output_dir=str(output_dir),
                ): tid
                for idx, tid in enumerate(task_ids)
            }
            for count, fut in enumerate(as_completed(futures), 1):
                result = fut.result()
                results.append(result)
                print(f"[{count}/{len(task_ids)}] {result['task_id']}: "
                      f"success={int(result['success'])} match={result.get('match_score','?')}")
        results.sort(key=lambda r: r["task_id"])

    summary = summarize(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump({"args": vars(args), "summary": summary, "results": results}, f, indent=2)

    # CSV
    if results:
        keys = ["task_id", "difficulty", "tier", "n_truth", "n_pred", "success",
                "reward", "match_score", "rms_ms", "logZ", "pred_periods_days"]
        with open(output_dir / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in results:
                row = dict(r)
                row["pred_periods_days"] = ";".join(str(x) for x in r["pred_periods_days"])
                w.writerow(row)

    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
