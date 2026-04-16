"""Mentor agent that reviews and guides the worker agent's actions."""

from __future__ import annotations

import os
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None


@dataclass
class MentorConfig:
    """Configuration for the mentor agent."""

    model: str = "gpt-5-mini"
    temperature: float = 0.2
    max_tokens_per_review: int = 2000
    max_mentor_calls: int = 5
    max_mentor_tokens: int = 50000
    intervention_policy: str = "key_decisions"  # "every_step" | "key_decisions" | "pre_submit_only"
    api_timeout_sec: float = 30.0


_MENTOR_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a senior exoplanet radial-velocity (RV) analyst mentoring a junior analyst.
    You see the worker's recent analysis steps and must provide brief, actionable guidance.

    ## Your Priorities
    1. **Period validation**: Check candidate periods against common aliases
       (1-day sampling alias, harmonics, baseline period). Flag suspicious periods.
    2. **Parameter completeness**: A full Keplerian fit requires 6 parameters:
       P (period), K (semi-amplitude), e (eccentricity), omega (argument of
       periastron), M0 (mean anomaly at reference epoch), gamma (systemic velocity).
       Flag if any are missing or defaulted.
    3. **l_rad conversion**: The canonical submission field is l_rad (mean longitude
       at t_ref = times_days[0]). Correct formula:
       l_rad = (Omega_rad + omega_rad + M0) % (2 * pi).
       Flag if the conversion looks wrong.
    4. **Fit quality**: Residual RMS should be close to median measurement uncertainty
       (median_sigma). If RMS >> median_sigma, the fit has not converged — suggest
       multi-start optimization or checking initial guesses.
    5. **Physical reasonableness**: Eccentricity in [0, 0.8], period > 0.5 days,
       m_sin_i > 0. Flag unphysical parameters.

    ## Response Rules
    - Keep responses under 3 sentences. Be specific with numbers.
    - Do NOT repeat or summarize the worker's analysis — only correct or redirect.
    - If the worker is on track, respond with exactly: LGTM
    - Focus on the single most important issue if multiple exist.
""")


class MentorPolicy:
    """Decides whether the mentor should intervene at a given point."""

    def __init__(self, policy: str = "key_decisions"):
        if policy not in ("every_step", "key_decisions", "pre_submit_only"):
            raise ValueError(
                f"Unknown mentor policy: {policy!r}. "
                f"Choose from: every_step, key_decisions, pre_submit_only"
            )
        self.policy = policy

    def should_intervene(
        self,
        event: str,
        tool_name: Optional[str] = None,
        tool_calls_made: int = 0,
        history: Optional[List[Dict]] = None,
        last_reward: Optional[float] = None,
    ) -> bool:
        """Return True if the mentor should review at this point.

        Events:
            post_tool    – after a tool execution completed
            pre_submit   – before submit_action is executed
            post_submit  – after a submission (reward available)
            stuck        – worker produced multiple text-only turns
        """
        if self.policy == "every_step":
            return event in ("post_tool", "pre_submit", "post_submit", "stuck")

        if self.policy == "pre_submit_only":
            return event == "pre_submit"

        # key_decisions (default)
        if event == "pre_submit":
            return True
        if event == "stuck":
            return True
        if event == "post_submit":
            # Intervene after a failed submission
            return last_reward is not None and last_reward < 0.5
        if event == "post_tool":
            # Intervene after early analysis steps to validate direction
            if tool_name == "PythonREPL" and tool_calls_made in (2, 3):
                return True
            return False
        return False


class MentorAgent:
    """Reviews worker agent actions and provides strategic guidance.

    The mentor maintains its own LLM client and token budget, separate
    from the worker agent.
    """

    def __init__(
        self,
        config: Optional[MentorConfig] = None,
        client: Optional[Any] = None,
    ):
        self.config = config or MentorConfig()
        self._provider: str = "openai_compat"
        self._api_model: str = self.config.model
        self.llm = self._initialize_client(client)
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._calls_made = 0

    def _initialize_client(self, client: Optional[Any]) -> Any:
        """Initialize API client based on model name."""
        if client is not None:
            return client

        model_lower = self.config.model.lower()

        if model_lower.startswith("openrouter/"):
            if openai is None:
                raise ImportError("openai package required for OpenRouter models.")
            routed_model = self.config.model.split("/", 1)[1].strip()
            api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Missing OPENROUTER_API_KEY for OpenRouter.")
            base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self._provider = "openai_compat"
            self._api_model = routed_model
            return openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.config.api_timeout_sec,
            )

        if "claude" in model_lower:
            if anthropic is None:
                raise ImportError("anthropic package required for Claude models.")
            self._provider = "anthropic"
            self._api_model = self.config.model
            return anthropic.Anthropic(timeout=self.config.api_timeout_sec)

        if "gpt" in model_lower or model_lower.startswith("o"):
            if openai is None:
                raise ImportError("openai package required for OpenAI models.")
            self._provider = "openai_compat"
            self._api_model = self.config.model
            return openai.OpenAI(timeout=self.config.api_timeout_sec)

        # Fallback: try OpenRouter if key is available
        if openai is not None and os.environ.get("OPENROUTER_API_KEY"):
            base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self._provider = "openai_compat"
            self._api_model = self.config.model
            return openai.OpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url=base_url,
                timeout=self.config.api_timeout_sec,
            )

        raise ValueError(
            f"Mentor model {self.config.model!r} not recognized. "
            f"Use 'claude-*', 'gpt-*', 'o*', or 'openrouter/<model>'."
        )

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def budget_exhausted(self) -> bool:
        return (
            self._calls_made >= self.config.max_mentor_calls
            or self.total_tokens >= self.config.max_mentor_tokens
        )

    def review(
        self,
        worker_messages: List[Dict[str, Any]],
        event: str,
        task_summary: Dict[str, Any],
    ) -> Optional[str]:
        """Generate mentor feedback for the worker's current state.

        Args:
            worker_messages: Recent conversation messages (last 3-4 turns).
            event: Trigger event type (post_tool, pre_submit, post_submit, stuck).
            task_summary: Dataset stats and context for the mentor.

        Returns:
            Guidance string, or None if the worker is on track / budget exhausted.
        """
        if self.budget_exhausted:
            return None

        # Build compact context for the mentor
        user_prompt = self._build_review_prompt(worker_messages, event, task_summary)

        try:
            if self._provider == "anthropic":
                response = self._call_anthropic(user_prompt)
            else:
                response = self._call_openai(user_prompt)
        except Exception as e:
            print(f"[Mentor] API error: {type(e).__name__}: {e}")
            return None

        self._calls_made += 1

        # If mentor says LGTM, no guidance needed
        if response and response.strip().upper().startswith("LGTM"):
            return None

        return response

    def _build_review_prompt(
        self,
        worker_messages: List[Dict[str, Any]],
        event: str,
        task_summary: Dict[str, Any],
    ) -> str:
        """Build a compact prompt summarizing the worker's state for mentor review."""
        parts = []

        # Task context
        parts.append(f"## Task Context")
        parts.append(f"- Observations: {task_summary.get('n_obs', '?')}")
        parts.append(f"- Time span: {task_summary.get('time_span_days', '?'):.1f} days")
        parts.append(f"- Median sigma: {task_summary.get('median_sigma', '?'):.3f} m/s")
        parts.append(f"- RV range: {task_summary.get('rv_min', '?'):.2f} to {task_summary.get('rv_max', '?'):.2f} m/s")
        parts.append(f"- Star mass: {task_summary.get('star_mass_sun', 1.0):.2f} M_sun")
        parts.append("")

        # Event context
        event_labels = {
            "post_tool": "After tool execution (early analysis)",
            "pre_submit": "Before submission — review parameters carefully",
            "post_submit": "After a failed submission — diagnose what went wrong",
            "stuck": "Worker appears stuck — provide direction",
        }
        parts.append(f"## Review Trigger: {event_labels.get(event, event)}")
        parts.append("")

        # Recent worker messages (compact)
        parts.append("## Recent Worker Activity")
        for msg in worker_messages[-6:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Handle Anthropic-style content blocks
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            text_parts.append(
                                f"[Tool call: {block.get('name', '?')}]"
                            )
                        elif block.get("type") == "tool_result":
                            result_text = block.get("content", "")
                            if len(result_text) > 500:
                                result_text = result_text[:500] + "...[truncated]"
                            text_parts.append(f"[Tool result]: {result_text}")
                content = "\n".join(text_parts)

            if isinstance(content, str) and len(content) > 800:
                content = content[:800] + "...[truncated]"

            # Handle OpenAI-style tool_calls in message
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                tc_summaries = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get("function", tc)
                        name = func.get("name", tc.get("name", "?"))
                        tc_summaries.append(f"[Tool call: {name}]")
                content = (content or "") + "\n" + "\n".join(tc_summaries)

            parts.append(f"**{role}**: {content}")
            parts.append("")

        return "\n".join(parts)

    def _call_anthropic(self, user_prompt: str) -> str:
        """Call Anthropic API for mentor review."""
        response = self.llm.messages.create(
            model=self._api_model,
            max_tokens=self.config.max_tokens_per_review,
            system=_MENTOR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.config.temperature,
        )
        self._total_input_tokens += getattr(response.usage, "input_tokens", 0)
        self._total_input_tokens += getattr(response.usage, "cache_creation_input_tokens", 0)
        self._total_input_tokens += getattr(response.usage, "cache_read_input_tokens", 0)
        self._total_output_tokens += getattr(response.usage, "output_tokens", 0)

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text
        return text

    def _call_openai(self, user_prompt: str) -> str:
        """Call OpenAI-compatible API for mentor review."""
        # gpt-5 series and o-series use max_completion_tokens instead of max_tokens
        model_lower = self._api_model.lower()
        use_completion_tokens = (
            "gpt-5" in model_lower
            or model_lower.startswith("o")
        )
        token_kwargs = (
            {"max_completion_tokens": self.config.max_tokens_per_review}
            if use_completion_tokens
            else {"max_tokens": self.config.max_tokens_per_review}
        )
        temp_kwargs = {}
        skip_temp = model_lower.startswith("o") or "gpt-5" in model_lower
        if not skip_temp:
            temp_kwargs["temperature"] = self.config.temperature

        response = self.llm.chat.completions.create(
            model=self._api_model,
            messages=[
                {"role": "system", "content": _MENTOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            **token_kwargs,
            **temp_kwargs,
        )
        self._total_input_tokens += getattr(response.usage, "prompt_tokens", 0) or 0
        self._total_output_tokens += getattr(response.usage, "completion_tokens", 0) or 0

        return response.choices[0].message.content or ""

    def get_usage_stats(self) -> Dict[str, Any]:
        """Return mentor usage statistics for trace output."""
        return {
            "mentor_model": self.config.model,
            "mentor_calls_made": self._calls_made,
            "mentor_input_tokens": self._total_input_tokens,
            "mentor_output_tokens": self._total_output_tokens,
            "mentor_total_tokens": self.total_tokens,
            "mentor_policy": self.config.intervention_policy,
        }
