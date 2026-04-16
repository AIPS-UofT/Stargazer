"""Factory for creating a TabularRvAgent with a mentor callback wired in."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .mentor import MentorAgent, MentorConfig, MentorPolicy
from .tabular_agent import TabularRvAgent, TabularAgentConfig


def _extract_task_summary(env) -> Dict[str, Any]:
    """Extract compact task summary from the environment for the mentor's context."""
    obs, _ = env.reset()
    times = np.asarray(obs["times_days"], dtype=float)
    rvs = np.asarray(obs["rvs_ms"], dtype=float)
    sigmas = np.asarray(obs["sigmas_ms"], dtype=float)
    meta = obs.get("meta", {})

    return {
        "n_obs": len(times),
        "time_span_days": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
        "median_sigma": float(np.median(sigmas)),
        "rv_min": float(rvs.min()),
        "rv_max": float(rvs.max()),
        "rv_std": float(rvs.std()),
        "star_mass_sun": float(meta.get("star_mass_sun", 1.0)),
        "task_id": meta.get("task_id", "unknown"),
    }


class _MentorCallback:
    """Callable that wires MentorAgent + MentorPolicy into the on_step_callback interface.

    Also exposes get_mentor_stats() so the trace can capture mentor usage.
    """

    def __init__(
        self,
        mentor: MentorAgent,
        policy: MentorPolicy,
        task_summary: Dict[str, Any],
    ):
        self._mentor = mentor
        self._policy = policy
        self._task_summary = task_summary

    def __call__(
        self,
        event: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict] = None,
        tool_result: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        tool_calls_made: int = 0,
        history: Optional[List[Dict]] = None,
        last_reward: Optional[float] = None,
    ) -> Optional[str]:
        """Invoked by TabularRvAgent at hook points."""
        if not self._policy.should_intervene(
            event=event,
            tool_name=tool_name,
            tool_calls_made=tool_calls_made,
            history=history,
            last_reward=last_reward,
        ):
            return None

        return self._mentor.review(
            worker_messages=messages or [],
            event=event,
            task_summary=self._task_summary,
        )

    def get_mentor_stats(self) -> Dict[str, Any]:
        return self._mentor.get_usage_stats()


def create_mentored_agent(
    env,
    worker_config: Optional[TabularAgentConfig] = None,
    mentor_config: Optional[MentorConfig] = None,
    worker_client: Optional[Any] = None,
    mentor_client: Optional[Any] = None,
    trace_dir=None,
) -> TabularRvAgent:
    """Create a TabularRvAgent with a mentor callback wired in.

    Args:
        env: The RvEnv environment instance.
        worker_config: Configuration for the worker agent.
        mentor_config: Configuration for the mentor agent.
        worker_client: Optional pre-initialized LLM client for the worker.
        mentor_client: Optional pre-initialized LLM client for the mentor.
        trace_dir: Optional directory for incremental trace output.

    Returns:
        A TabularRvAgent instance with the mentor callback attached.
    """
    mentor_config = mentor_config or MentorConfig()
    mentor = MentorAgent(config=mentor_config, client=mentor_client)
    policy = MentorPolicy(policy=mentor_config.intervention_policy)

    # Extract task summary — note: env.reset() is called here but the agent's
    # run() will call reset() again, so this is safe.
    task_summary = _extract_task_summary(env)

    callback = _MentorCallback(mentor=mentor, policy=policy, task_summary=task_summary)

    return TabularRvAgent(
        env=env,
        config=worker_config,
        client=worker_client,
        on_step_callback=callback,
        trace_dir=trace_dir,
    )
