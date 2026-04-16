from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import inspect

from .config import Task
from .evaluator import evaluate_submission
from .bank import _apply_rv_only_compat

class RvEnv:
    VALID_MODES = {"model_only","params_only","params_and_model"}
    VALID_COMPONENTS = {"likelihood","delta_bic","neg_rms","match","count"}

    def __init__(self,
                 task: Optional[Task] = None,
                 task_sampler = None,
                 submission_mode: str = "params_and_model",
                 reward_weights: Dict[str, float] = None,
                 success_criteria: Optional[Dict[str, float]] = None,
                 max_steps: int = 1):
        if submission_mode not in self.VALID_MODES:
            raise ValueError(f"Invalid submission_mode: '{submission_mode}'. Must be one of {self.VALID_MODES}")
        self._task = task
        self._task_sampler = task_sampler
        self.submission_mode = submission_mode
        self.reward_weights = reward_weights or {"likelihood": 1.0, "delta_bic": 0.3, "neg_rms": 0.1, "match": 1.0, "count": 0.2}
        self.success_criteria = success_criteria or {
            # Model-fit success (used when mode includes a model): reward-relevant BIC improvement and noise-like residuals.
            "min_delta_bic_per_point": 0.0,
            "max_rms_factor_over_median_sigma": 1.5,
            # Parameter-recovery success (used when mode includes planets): close match and correct planet count.
            "min_match_score": 0.8,
            "require_count_match": 1.0,
        }
        unknown = set(self.reward_weights.keys()) - self.VALID_COMPONENTS
        if unknown:
            raise ValueError(f"Unknown reward component(s): {unknown}. Valid: {self.VALID_COMPONENTS}")
        self.max_steps = max_steps
        self._steps = 0

    @property
    def task(self) -> Task:
        if self._task is None:
            raise RuntimeError("No task loaded. Call reset() with a sampler or provide a Task.")
        return self._task

    def reset(self, *, seed: Optional[int] = None) -> Tuple[Dict[str,Any], Dict[str,Any]]:
        self._steps = 0
        if self._task_sampler is not None:
            # Try to pass seed to task_sampler if it accepts it
            # This supports TaskFactory.sample(seed=...) and similar callables
            try:
                sig = inspect.signature(self._task_sampler)
                if 'seed' in sig.parameters:
                    self._task = self._task_sampler(seed=seed)
                else:
                    self._task = self._task_sampler()
            except (ValueError, TypeError):
                # If signature inspection fails, try calling with seed first, then without
                try:
                    self._task = self._task_sampler(seed=seed) if seed is not None else self._task_sampler()
                except TypeError:
                    self._task = self._task_sampler()
        if self._task is None:
            raise RuntimeError("RvEnv.reset() requires a task or a sampler that returns a Task.")
        # Convert to RV-only semantics (handles REBOUND l_rad/Omega_rad mismatch)
        self._task = _apply_rv_only_compat(self._task)
        obs = self._make_observation(self._task)
        info = {"task_id": self._task.task_id, "truth_difficulty": self._task.truth_difficulty, "difficulty_details": self._task.difficulty_details}
        return obs, info

    def _make_observation(self, task: Task) -> Dict[str,Any]:
        meta = {
            "time_unit": "day",
            "rv_unit": "m/s",
            "instrument_labels": [inst.label for inst in task.config.instruments],
            "star_mass_sun": float(task.config.star.M_star_sun),
            "los_axis": str(task.config.los_axis),
            "integrator_preference": str(task.config.integrator_preference),
            "engine": str(task.config.engine),
        }
        # Pass through task description and hints if present
        if task.meta.get("task_description"):
            meta["task_description"] = task.meta["task_description"]
        if task.meta.get("hints"):
            meta["hints"] = task.meta["hints"]
        if task.meta.get("reference"):
            meta["reference"] = task.meta["reference"]

        return {
            "times_days": task.observations.times_days,
            "rvs_ms": task.observations.rvs_ms,
            "sigmas_ms": task.observations.sigmas_ms,
            "instruments": task.observations.instruments,
            "meta": meta,
        }

    def _evaluate_success(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        criteria = self.success_criteria
        components = metrics.get("components", {}) if isinstance(metrics.get("components"), dict) else {}
        residuals = metrics.get("residuals", {}) if isinstance(metrics.get("residuals"), dict) else {}
        task_hints = self.task.meta.get("hints", {}) if isinstance(self.task.meta, dict) else {}

        def _hint_float(key: str) -> Optional[float]:
            if not isinstance(task_hints, dict):
                return None
            raw = task_hints.get(key)
            if raw is None:
                return None
            try:
                val = float(raw)
            except (TypeError, ValueError):
                return None
            return val if np.isfinite(val) else None

        details: Dict[str, Any] = {}
        hints: List[str] = []
        success = True
        ok_rms = False

        median_sigma = float(np.median(np.asarray(self.task.observations.sigmas_ms, dtype=float)))
        details["median_sigma_ms"] = median_sigma

        if self.submission_mode in ("params_and_model", "model_only"):
            delta_bic_per_point = components.get("delta_bic", None)
            rms = residuals.get("rms", None)
            min_db = float(criteria.get("min_delta_bic_per_point", 0.0))
            max_factor = float(criteria.get("max_rms_factor_over_median_sigma", 1.5))
            max_rms_hint = _hint_float("max_rms_ms")
            max_rms_ms = (
                float(max_rms_hint)
                if (max_rms_hint is not None and max_rms_hint > 0.0)
                else float(max_factor * median_sigma)
            )
            ok_db = (delta_bic_per_point is not None) and (float(delta_bic_per_point) > min_db)
            ok_rms = (rms is not None) and (float(rms) <= max_rms_ms)
            details.update(
                {
                    "delta_bic_per_point": float(delta_bic_per_point) if delta_bic_per_point is not None else None,
                    "rms_ms": float(rms) if rms is not None else None,
                    "ok_delta_bic": bool(ok_db),
                    "ok_rms": bool(ok_rms),
                    "min_delta_bic_per_point": min_db,
                    "max_rms_ms": max_rms_ms,
                }
            )
            success = success and ok_db and ok_rms

        if self.submission_mode in ("params_and_model", "params_only"):
            match_score = components.get("match", None)
            count_term = components.get("count", None)
            min_match_hint = _hint_float("target_match_score")
            min_match = (
                float(min_match_hint)
                if (min_match_hint is not None and min_match_hint >= 0.0)
                else float(criteria.get("min_match_score", 0.8))
            )
            require_count_match = bool(float(criteria.get("require_count_match", 1.0)) >= 0.5)
            ok_match = (match_score is not None) and (float(match_score) >= min_match)
            ok_count = True
            if require_count_match:
                ok_count = (count_term is not None) and (float(count_term) == 0.0)
            details.update(
                {
                    "match_score": float(match_score) if match_score is not None else None,
                    "count_term": float(count_term) if count_term is not None else None,
                    "ok_match": bool(ok_match),
                    "ok_count": bool(ok_count),
                    "min_match_score": min_match,
                    "require_count_match": require_count_match,
                }
            )
            success = success and ok_match and ok_count
            if ok_rms and (not ok_match):
                hints.append(
                    "你的拟合曲线非常完美，但物理参数与坐标系不匹配，请检查你的相位 $l_{rad}$ 转换公式。"
                )

        return {"success": bool(success), "details": details, "hints": hints}

    def step(self, action: Dict[str,Any]) -> Tuple[Dict[str,Any], float, bool, Dict[str,Any]]:
        if self._task is None:
            raise RuntimeError("Call reset() before step().")
        self._steps += 1
        reward, metrics = evaluate_submission(
            config=self._task.config,
            obs=self._task.observations,
            submission=action,
            truth_planets=self._task.config.planets,
            reward_weights=self.reward_weights,
            mode=self.submission_mode
        )
        obs = self._make_observation(self._task)  # unchanged (no active scheduling)
        success_eval = self._evaluate_success(metrics)
        # Stop if max steps reached OR success achieved
        done = bool(self._steps >= self.max_steps or success_eval["success"])
        info = {
            "task_id": self._task.task_id,
            "metrics": metrics,
            "success": success_eval["success"],
            "success_details": success_eval["details"],
            "hints": success_eval.get("hints", []),
        }
        return obs, float(reward), done, info
