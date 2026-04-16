from __future__ import annotations

from typing import Dict, Any, List, Optional
from html import escape
from collections import Counter
import json
import re


def convert_trace_to_json(trace: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert agent trace to JSON-serializable format.

    Args:
        trace: Raw trace from agent.run()

    Returns:
        JSON-serializable trace dictionary
    """
    intermediate_steps = []
    for step in trace.get("intermediate_steps", []):
        tool_info = {
            "tool": step[0].tool,
            "tool_input": step[0].tool_input,
            "message_log": (
                [{"content": msg.content} for msg in step[0].message_log]
                if hasattr(step[0], "message_log") and step[0].message_log
                else None
            ),
        }
        tool_output = step[1]
        intermediate_steps.append({"tool_info": tool_info, "tool_output": tool_output})

    json_data = {
        "input": trace.get("input"),
        "output": trace.get("output"),
        "intermediate_steps": intermediate_steps,
        "input_tokens_used": trace.get("input_tokens_used", 0),
        "output_tokens_used": trace.get("output_tokens_used", 0),
        "error_message": trace.get("error_message"),
        "stop_reason": trace.get("stop_reason"),
        "history": trace.get("history", []),
        # Extended metadata
        "metadata": trace.get("metadata", {}),
        "execution_time_seconds": trace.get("execution_time_seconds"),
        "task_info": trace.get("task_info", {}),
        "config": trace.get("config", {}),
    }

    return json_data


def _compute_statistics(trace: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute detailed statistics from the trace.

    Returns:
        Dictionary with comprehensive statistics
    """
    steps = trace.get("intermediate_steps", [])
    history = trace.get("history", [])

    # Tool usage statistics
    tool_counts = Counter()
    tool_input_lengths = {}
    tool_output_lengths = {}

    for step in steps:
        tool_info = step.get("tool_info", {})
        tool_name = tool_info.get("tool", "Unknown")
        tool_counts[tool_name] += 1

        # Track input/output lengths
        input_len = len(str(tool_info.get("tool_input", "")))
        output_len = len(str(step.get("tool_output", "")))

        if tool_name not in tool_input_lengths:
            tool_input_lengths[tool_name] = []
            tool_output_lengths[tool_name] = []
        tool_input_lengths[tool_name].append(input_len)
        tool_output_lengths[tool_name].append(output_len)

    # Token efficiency
    total_tokens = trace.get("input_tokens_used", 0) + trace.get("output_tokens_used", 0)
    tokens_per_step = total_tokens / len(steps) if steps else 0

    # Submission analysis
    rewards = [h.get("reward", 0) for h in history]
    best_reward = max(rewards) if rewards else None
    worst_reward = min(rewards) if rewards else None
    reward_improvement = rewards[-1] - rewards[0] if len(rewards) > 1 else 0

    # Success detection: prefer environment correctness signal when available.
    final_success = False
    if history:
        if "success" in history[-1]:
            final_success = bool(history[-1].get("success", False))
        else:
            # Backward compatibility for older traces.
            final_success = bool(history[-1].get("done", False))

    return {
        "tool_usage": {
            "counts": dict(tool_counts),
            "total_tool_calls": sum(tool_counts.values()),
            "unique_tools_used": len(tool_counts),
            "avg_input_length": {k: sum(v)/len(v) for k, v in tool_input_lengths.items()},
            "avg_output_length": {k: sum(v)/len(v) for k, v in tool_output_lengths.items()},
        },
        "token_efficiency": {
            "total_tokens": total_tokens,
            "tokens_per_step": round(tokens_per_step, 2),
            "input_ratio": round(trace.get("input_tokens_used", 0) / total_tokens, 3) if total_tokens else 0,
            "output_ratio": round(trace.get("output_tokens_used", 0) / total_tokens, 3) if total_tokens else 0,
        },
        "submission_analysis": {
            "total_submissions": len(history),
            "best_reward": best_reward,
            "worst_reward": worst_reward,
            "final_reward": rewards[-1] if rewards else None,
            "reward_improvement": round(reward_improvement, 4) if reward_improvement else 0,
            "success": final_success,
        },
        "execution": {
            "total_steps": len(steps),
            "stop_reason": trace.get("stop_reason"),
            "had_error": bool(trace.get("error_message")),
        }
    }


def _extract_detected_planets(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract detected planet parameters from the trace.

    Returns:
        List of detected planet dictionaries
    """
    history = trace.get("history", [])
    if not history:
        return []

    # Get the last submission's metrics
    last_entry = history[-1]
    metrics = last_entry.get("metrics", {})

    planets = []
    submission = metrics.get("submission", {})
    if isinstance(submission, dict) and "planets" in submission:
        for i, planet in enumerate(submission.get("planets", [])):
            planets.append({
                "planet_id": i + 1,
                "period_days": planet.get("period_days"),
                "semi_amplitude_ms": planet.get("semi_amplitude_ms"),
                "eccentricity": planet.get("eccentricity"),
                "omega_rad": planet.get("omega_rad"),
                "phase_rad": planet.get("phase_rad"),
            })

    return planets


def _extract_analysis_timeline(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract a timeline of agent analysis activities.

    Returns:
        List of analysis events with descriptions
    """
    steps = trace.get("intermediate_steps", [])
    timeline = []

    for i, step in enumerate(steps):
        tool_info = step.get("tool_info", {})
        tool_name = tool_info.get("tool", "Unknown")
        tool_input = str(tool_info.get("tool_input", ""))
        tool_output = str(step.get("tool_output", ""))

        event = {
            "step": i + 1,
            "tool": tool_name,
            "action_type": _classify_action(tool_name, tool_input),
        }

        # Extract key findings from outputs
        if "periodogram" in tool_input.lower() or "lombscargle" in tool_input.lower():
            event["action_type"] = "periodogram_analysis"
            # Try to extract peak periods
            periods = re.findall(r"period[s]?\s*[:=]\s*([\d.]+)", tool_output.lower())
            if periods:
                event["key_findings"] = {"candidate_periods": periods[:5]}

        elif "fit" in tool_input.lower() or "optimize" in tool_input.lower():
            event["action_type"] = "orbital_fitting"
            # Try to extract RMS
            rms_match = re.search(r"rms\s*[:=]\s*([\d.]+)", tool_output.lower())
            if rms_match:
                event["key_findings"] = {"residual_rms": rms_match.group(1)}

        elif tool_name == "submit_action":
            event["action_type"] = "submission"
            # Get reward if available
            history = trace.get("history", [])
            if history:
                for h in history:
                    if h.get("step") == len(timeline) + 1 or i == len(steps) - 1:
                        event["key_findings"] = {"reward": h.get("reward")}
                        break

        timeline.append(event)

    return timeline


def _classify_action(tool_name: str, tool_input: str) -> str:
    """Classify the type of action based on tool and input."""
    input_lower = tool_input.lower()

    if tool_name == "submit_action":
        return "submission"
    elif tool_name == "Agent Error":
        return "error"
    elif tool_name == "Assistant Text":
        return "reasoning"

    # Classify PythonREPL actions
    if "periodogram" in input_lower or "lombscargle" in input_lower:
        return "periodogram_analysis"
    elif "fit" in input_lower or "optimize" in input_lower or "least_squares" in input_lower:
        return "orbital_fitting"
    elif "residual" in input_lower:
        return "residual_analysis"
    elif "print" in input_lower and ("times" in input_lower or "rvs" in input_lower):
        return "data_exploration"
    elif "import" in input_lower:
        return "setup"
    else:
        return "computation"


def trace_to_markdown(trace: Dict[str, Any]) -> str:
    """
    Convert trace to markdown format for easy viewing.

    Args:
        trace: Agent execution trace

    Returns:
        Markdown-formatted trace with comprehensive analysis
    """
    md = ["#  Agent Execution Trace\n"]

    # Compute statistics
    stats = _compute_statistics(trace)
    planets = _extract_detected_planets(trace)
    timeline = _extract_analysis_timeline(trace)

    # ==================== Executive Summary ====================
    md.append("##  Executive Summary\n")

    # Success/Failure banner
    success = stats["submission_analysis"]["success"]
    final_reward = stats["submission_analysis"]["final_reward"]
    if success:
        md.append(f"###  Task Completed Successfully\n")
    elif final_reward is not None:
        md.append(f"###  Task Incomplete\n")
    else:
        md.append(f"###  No Submission Made\n")

    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| **Final Reward** | {final_reward if final_reward is not None else 'N/A'} |")
    md.append(f"| **Best Reward** | {stats['submission_analysis']['best_reward'] if stats['submission_analysis']['best_reward'] is not None else 'N/A'} |")
    md.append(f"| **Total Steps** | {stats['execution']['total_steps']} |")
    md.append(f"| **Submissions** | {stats['submission_analysis']['total_submissions']} |")
    md.append(f"| **Stop Reason** | {stats['execution']['stop_reason'] or 'N/A'} |")
    md.append("")

    # ==================== Token Usage ====================
    md.append("##  Token Usage\n")
    md.append("| Category | Tokens | Percentage |")
    md.append("|----------|--------|------------|")
    md.append(f"| Input | {trace.get('input_tokens_used', 0):,} | {stats['token_efficiency']['input_ratio']*100:.1f}% |")
    md.append(f"| Output | {trace.get('output_tokens_used', 0):,} | {stats['token_efficiency']['output_ratio']*100:.1f}% |")
    md.append(f"| **Total** | **{stats['token_efficiency']['total_tokens']:,}** | 100% |")
    md.append("")
    md.append(f" **Efficiency**: {stats['token_efficiency']['tokens_per_step']:.0f} tokens/step\n")

    # ==================== Tool Usage Statistics ====================
    md.append("##  Tool Usage Statistics\n")
    tool_usage = stats["tool_usage"]

    if tool_usage["counts"]:
        md.append("| Tool | Call Count | Avg Input Len | Avg Output Len |")
        md.append("|------|------------|---------------|----------------|")
        for tool_name, count in sorted(tool_usage["counts"].items(), key=lambda x: -x[1]):
            avg_in = tool_usage["avg_input_length"].get(tool_name, 0)
            avg_out = tool_usage["avg_output_length"].get(tool_name, 0)
            md.append(f"| {tool_name} | {count} | {avg_in:.0f} | {avg_out:.0f} |")
        md.append("")

    # ==================== Detected Planets ====================
    if planets:
        md.append("##  Detected Planets\n")
        md.append("| # | Period (days) | K (m/s) | Eccentricity | ω (rad) | Phase (rad) |")
        md.append("|---|---------------|---------|--------------|---------|-------------|")
        for p in planets:
            md.append(
                f"| {p['planet_id']} | "
                f"{p['period_days']:.4f} | "
                f"{p['semi_amplitude_ms']:.3f} | "
                f"{p['eccentricity']:.4f} | "
                f"{p['omega_rad']:.4f} | "
                f"{p['phase_rad']:.4f} |"
            )
        md.append("")

    # ==================== Analysis Timeline ====================
    md.append("##  Analysis Timeline\n")

    # Group by action type
    action_summary = Counter(e["action_type"] for e in timeline)
    md.append("### Action Distribution\n")
    for action, count in action_summary.most_common():
        emoji = {
            "periodogram_analysis": "",
            "orbital_fitting": "",
            "submission": "",
            "data_exploration": "",
            "residual_analysis": "",
            "computation": "",
            "reasoning": "",
            "setup": "",
            "error": "",
        }.get(action, "•")
        md.append(f"- {emoji} **{action}**: {count} times")
    md.append("")

    # Timeline details
    md.append("### Step-by-Step Timeline\n")
    md.append("| Step | Action Type | Key Findings |")
    md.append("|------|-------------|--------------|")
    for event in timeline[:20]:  # Limit to first 20 for readability
        findings = event.get("key_findings", {})
        findings_str = ", ".join(f"{k}={v}" for k, v in findings.items()) if findings else "-"
        md.append(f"| {event['step']} | {event['action_type']} | {findings_str} |")
    if len(timeline) > 20:
        md.append(f"| ... | *{len(timeline) - 20} more steps* | - |")
    md.append("")

    # ==================== Submission History ====================
    if trace.get("history"):
        md.append("##  Submission History\n")
        md.append("| # | Reward | Done | Improvement |")
        md.append("|---|--------|------|-------------|")
        prev_reward = None
        for entry in trace["history"]:
            reward = entry.get("reward", 0)
            improvement = ""
            if prev_reward is not None:
                diff = reward - prev_reward
                if diff > 0:
                    improvement = f" +{diff:.4f}"
                elif diff < 0:
                    improvement = f" {diff:.4f}"
                else:
                    improvement = " 0"
            done_emoji = "" if entry.get("done", False) else ""
            md.append(f"| {entry.get('step', '?')} | {reward:.4f} | {done_emoji} | {improvement} |")
            prev_reward = reward
        md.append("")

        # Reward progression summary
        rewards = [h.get("reward", 0) for h in trace["history"]]
        if len(rewards) > 1:
            md.append(f" **Reward Progression**: {rewards[0]:.4f} → {rewards[-1]:.4f} ")
            improvement = rewards[-1] - rewards[0]
            if improvement > 0:
                md.append(f"(improved by +{improvement:.4f})\n")
            elif improvement < 0:
                md.append(f"(decreased by {improvement:.4f})\n")
            else:
                md.append("(no change)\n")

    # ==================== Error Section ====================
    if trace.get("error_message"):
        md.append("##  Error Details\n")
        md.append("```")
        md.append(trace['error_message'][:1000])
        if len(trace['error_message']) > 1000:
            md.append("...(truncated)")
        md.append("```\n")

    # ==================== Configuration ====================
    if trace.get("config"):
        md.append("##  Agent Configuration\n")
        config = trace.get("config", {})
        md.append("| Parameter | Value |")
        md.append("|-----------|-------|")
        for key, value in config.items():
            md.append(f"| {key} | {value} |")
        md.append("")

    # ==================== Task Information ====================
    if trace.get("task_info"):
        md.append("##  Task Information\n")
        task_info = trace.get("task_info", {})
        for key, value in task_info.items():
            md.append(f"- **{key}**: {value}")
        md.append("")

    # ==================== Input Prompt Preview ====================
    if trace.get("input"):
        md.append("##  System Prompt Preview\n")
        input_text = trace['input']
        md.append("<details>")
        md.append("<summary>Click to expand (first 800 characters)</summary>\n")
        md.append("```")
        md.append(input_text[:800])
        if len(input_text) > 800:
            md.append("...(truncated)")
        md.append("```")
        md.append("</details>\n")

    # ==================== Detailed Steps ====================
    md.append("##  Detailed Execution Steps\n")
    md.append("<details>")
    md.append("<summary>Click to expand all steps</summary>\n")

    for i, step in enumerate(trace.get("intermediate_steps", []), 1):
        tool_info = step.get("tool_info", {})
        tool_name = tool_info.get("tool", "Unknown")

        # Choose emoji based on tool
        tool_emoji = {
            "PythonREPL": "",
            "submit_action": "",
            "Agent Error": "",
            "Assistant Text": "",
        }.get(tool_name, "")

        md.append(f"### Step {i}: {tool_emoji} {tool_name}\n")

        if tool_info.get("message_log"):
            md.append("** Agent Message:**\n")
            for msg in tool_info["message_log"]:
                content = msg.get('content', '')[:600]
                if len(msg.get('content', '')) > 600:
                    content += "...(truncated)"
                md.append(f"> {content}\n")

        tool_input = tool_info.get('tool_input', '')
        if isinstance(tool_input, dict):
            tool_input = json.dumps(tool_input, indent=2)
        md.append("** Tool Input:**\n")
        md.append(f"```python\n{str(tool_input)[:800]}")
        if len(str(tool_input)) > 800:
            md.append("...(truncated)")
        md.append("```\n")

        md.append("** Tool Output:**\n")
        output = str(step.get("tool_output", ""))
        if len(output) > 600:
            output = output[:600] + "...(truncated)"
        md.append(f"```\n{output}\n```\n")

        md.append("---\n")

    md.append("</details>\n")

    # ==================== Final Output ====================
    if trace.get("output"):
        md.append("##  Final Output\n")
        output = trace['output']
        if isinstance(output, dict):
            md.append("```json")
            md.append(json.dumps(output, indent=2))
            md.append("```\n")
        else:
            md.append(f"```\n{output}\n```\n")

    # ==================== Footer ====================
    md.append("---")
    md.append("*Generated by Stargazer Agent Execution Tracer*\n")

    return "\n".join(md)


def safe_escape(s):
    """Safely escape a value that might be a string or another type."""
    if isinstance(s, str):
        return escape(s)
    return escape(str(s))


def trace_to_html(trace: Dict[str, Any]) -> str:
    """
    Convert trace to HTML format for rich viewing with comprehensive visualizations.

    Args:
        trace: Agent execution trace

    Returns:
        HTML-formatted trace with charts, statistics, and interactive elements
    """
    # Compute statistics
    stats = _compute_statistics(trace)
    planets = _extract_detected_planets(trace)
    timeline = _extract_analysis_timeline(trace)
    history = trace.get("history", [])

    # Build steps HTML
    steps_html = []
    for i, step in enumerate(trace.get("intermediate_steps", []), 1):
        tool_info = step.get("tool_info", {})
        tool_name = tool_info.get("tool", "Unknown")
        tool_input = tool_info.get("tool_input", "")
        if isinstance(tool_input, dict):
            tool_input = json.dumps(tool_input, indent=2)
        tool_input = str(tool_input)
        tool_output = str(step.get("tool_output", ""))

        # Tool-specific styling
        tool_class = "step"
        tool_icon = ""
        if tool_name == "PythonREPL":
            tool_class = "step python-step"
            tool_icon = ""
        elif tool_name == "submit_action":
            tool_class = "step submit-step"
            tool_icon = ""
        elif tool_name == "Agent Error":
            tool_class = "step error-step"
            tool_icon = ""
        elif tool_name == "Assistant Text":
            tool_class = "step text-step"
            tool_icon = ""

        message_html = ""
        if tool_info.get("message_log"):
            messages = [msg.get("content", "") for msg in tool_info["message_log"]]
            message_content = " ".join(messages)[:800]
            if len(" ".join(messages)) > 800:
                message_content += "...(truncated)"
            message_html = f'<div class="message"><strong> Agent Message:</strong><pre>{safe_escape(message_content)}</pre></div>'

        steps_html.append(
            f"""
            <div class="{tool_class}">
                <div class="step-header">
                    <span class="step-number">Step {i}</span>
                    <span class="step-tool">{tool_icon} {safe_escape(tool_name)}</span>
                </div>
                {message_html}
                <details class="tool-details">
                    <summary> Tool Input</summary>
                    <div class="tool-input">
                        <pre><code>{safe_escape(tool_input[:1500])}</code></pre>
                    </div>
                </details>
                <details class="tool-details" open>
                    <summary> Tool Output</summary>
                    <div class="tool-output">
                        <pre>{safe_escape(tool_output[:1500])}</pre>
                    </div>
                </details>
            </div>
            """
        )

    # Build planets table HTML
    planets_html = ""
    if planets:
        planets_rows = ""
        for p in planets:
            planets_rows += f"""
            <tr>
                <td>{p['planet_id']}</td>
                <td>{p['period_days']:.4f}</td>
                <td>{p['semi_amplitude_ms']:.3f}</td>
                <td>{p['eccentricity']:.4f}</td>
                <td>{p['omega_rad']:.4f}</td>
                <td>{p['phase_rad']:.4f}</td>
            </tr>
            """
        planets_html = f"""
        <div class="section planets-section">
            <h2> Detected Planets</h2>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Period (days)</th>
                        <th>K (m/s)</th>
                        <th>Eccentricity</th>
                        <th>ω (rad)</th>
                        <th>Phase (rad)</th>
                    </tr>
                </thead>
                <tbody>
                    {planets_rows}
                </tbody>
            </table>
        </div>
        """

    # Build tool usage chart data
    tool_counts = stats["tool_usage"]["counts"]
    tool_chart_data = json.dumps(list(tool_counts.items()))

    # Build submission history chart data
    reward_data = [[i+1, h.get("reward", 0)] for i, h in enumerate(history)]
    reward_chart_data = json.dumps(reward_data)

    # Build timeline HTML
    timeline_html = ""
    action_colors = {
        "periodogram_analysis": "#3498db",
        "orbital_fitting": "#e74c3c",
        "submission": "#2ecc71",
        "data_exploration": "#9b59b6",
        "residual_analysis": "#f39c12",
        "computation": "#95a5a6",
        "reasoning": "#1abc9c",
        "setup": "#34495e",
        "error": "#c0392b",
    }
    for event in timeline[:30]:
        color = action_colors.get(event["action_type"], "#95a5a6")
        timeline_html += f"""
        <div class="timeline-item" style="border-left-color: {color};">
            <span class="timeline-step">Step {event['step']}</span>
            <span class="timeline-action" style="background: {color};">{event['action_type']}</span>
        </div>
        """

    # Success status
    success = stats["submission_analysis"]["success"]
    final_reward = stats["submission_analysis"]["final_reward"]
    status_class = "success" if success else "incomplete"
    status_text = " Task Completed Successfully" if success else " Task Incomplete"
    status_icon = "" if success else ""

    # Build submission history table
    history_html = ""
    if history:
        history_rows = ""
        prev_reward = None
        for entry in history:
            reward = entry.get("reward", 0)
            done = entry.get("done", False)
            improvement = ""
            if prev_reward is not None:
                diff = reward - prev_reward
                if diff > 0:
                    improvement = f'<span class="improvement positive">+{diff:.4f}</span>'
                elif diff < 0:
                    improvement = f'<span class="improvement negative">{diff:.4f}</span>'
                else:
                    improvement = '<span class="improvement neutral">0</span>'
            done_badge = '<span class="badge success">Done</span>' if done else '<span class="badge pending">Ongoing</span>'
            history_rows += f"""
            <tr>
                <td>{entry.get('step', '?')}</td>
                <td><strong>{reward:.4f}</strong></td>
                <td>{done_badge}</td>
                <td>{improvement}</td>
            </tr>
            """
            prev_reward = reward
        history_html = f"""
        <div class="section">
            <h2> Submission History</h2>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Step</th>
                        <th>Reward</th>
                        <th>Status</th>
                        <th>Change</th>
                    </tr>
                </thead>
                <tbody>
                    {history_rows}
                </tbody>
            </table>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title> Agent Execution Trace</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            :root {{
                --primary: #2563eb;
                --success: #10b981;
                --warning: #f59e0b;
                --danger: #ef4444;
                --gray-50: #f9fafb;
                --gray-100: #f3f4f6;
                --gray-200: #e5e7eb;
                --gray-300: #d1d5db;
                --gray-700: #374151;
                --gray-900: #111827;
            }}

            * {{
                box-sizing: border-box;
            }}

            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                background: var(--gray-50);
                color: var(--gray-900);
                line-height: 1.6;
            }}

            h1 {{
                text-align: center;
                color: var(--gray-900);
                margin-bottom: 30px;
                font-size: 2em;
            }}

            h2 {{
                color: var(--gray-700);
                border-bottom: 2px solid var(--gray-200);
                padding-bottom: 10px;
                margin-top: 30px;
            }}

            /* Status Banner */
            .status-banner {{
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 30px;
                text-align: center;
                font-size: 1.3em;
            }}
            .status-banner.success {{
                background: linear-gradient(135deg, #d1fae5, #a7f3d0);
                border: 2px solid var(--success);
            }}
            .status-banner.incomplete {{
                background: linear-gradient(135deg, #fef3c7, #fde68a);
                border: 2px solid var(--warning);
            }}

            /* Dashboard Grid */
            .dashboard {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}

            .metric-card {{
                background: white;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                text-align: center;
            }}

            .metric-value {{
                font-size: 2.5em;
                font-weight: bold;
                color: var(--primary);
            }}

            .metric-label {{
                color: var(--gray-700);
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}

            /* Token Breakdown */
            .token-bar {{
                height: 30px;
                border-radius: 15px;
                overflow: hidden;
                display: flex;
                margin: 15px 0;
            }}

            .token-input {{
                background: #3b82f6;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 0.8em;
            }}

            .token-output {{
                background: #10b981;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 0.8em;
            }}

            /* Sections */
            .section {{
                background: white;
                padding: 25px;
                border-radius: 12px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                margin-bottom: 25px;
            }}

            /* Data Tables */
            .data-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}

            .data-table th, .data-table td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid var(--gray-200);
            }}

            .data-table th {{
                background: var(--gray-100);
                font-weight: 600;
                color: var(--gray-700);
            }}

            .data-table tr:hover {{
                background: var(--gray-50);
            }}

            /* Badges */
            .badge {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.8em;
                font-weight: 500;
            }}

            .badge.success {{
                background: #d1fae5;
                color: #065f46;
            }}

            .badge.pending {{
                background: #fef3c7;
                color: #92400e;
            }}

            /* Improvement indicators */
            .improvement {{
                font-weight: 600;
            }}
            .improvement.positive {{
                color: var(--success);
            }}
            .improvement.negative {{
                color: var(--danger);
            }}
            .improvement.neutral {{
                color: var(--gray-700);
            }}

            /* Timeline */
            .timeline {{
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                padding: 15px 0;
            }}

            .timeline-item {{
                padding: 8px 12px;
                background: white;
                border-radius: 8px;
                border-left: 4px solid var(--primary);
                font-size: 0.85em;
                box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            }}

            .timeline-step {{
                color: var(--gray-700);
                margin-right: 8px;
            }}

            .timeline-action {{
                display: inline-block;
                padding: 2px 8px;
                border-radius: 4px;
                color: white;
                font-size: 0.8em;
            }}

            /* Tool Usage */
            .tool-usage-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }}

            .tool-card {{
                background: var(--gray-50);
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }}

            .tool-count {{
                font-size: 2em;
                font-weight: bold;
                color: var(--primary);
            }}

            .tool-name {{
                color: var(--gray-700);
                font-size: 0.9em;
            }}

            /* Steps */
            .step {{
                background: white;
                padding: 20px;
                margin-bottom: 15px;
                border-radius: 12px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                border-left: 4px solid var(--gray-300);
            }}

            .step.python-step {{
                border-left-color: #3776ab;
            }}

            .step.submit-step {{
                border-left-color: var(--success);
            }}

            .step.error-step {{
                border-left-color: var(--danger);
                background: #fef2f2;
            }}

            .step.text-step {{
                border-left-color: #9333ea;
            }}

            .step-header {{
                display: flex;
                align-items: center;
                gap: 15px;
                margin-bottom: 15px;
            }}

            .step-number {{
                background: var(--gray-200);
                padding: 4px 12px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 0.85em;
            }}

            .step-tool {{
                font-weight: 600;
                color: var(--gray-700);
            }}

            .message {{
                background: #fef3c7;
                padding: 15px;
                margin: 15px 0;
                border-radius: 8px;
                border-left: 3px solid var(--warning);
            }}

            .tool-details {{
                margin: 10px 0;
            }}

            .tool-details summary {{
                cursor: pointer;
                padding: 8px 12px;
                background: var(--gray-100);
                border-radius: 6px;
                font-weight: 500;
                color: var(--gray-700);
            }}

            .tool-details summary:hover {{
                background: var(--gray-200);
            }}

            .tool-input {{
                background: var(--gray-50);
                padding: 15px;
                margin-top: 10px;
                border-radius: 8px;
                border-left: 3px solid var(--primary);
            }}

            .tool-output {{
                background: #f0fdf4;
                padding: 15px;
                margin-top: 10px;
                border-radius: 8px;
                border-left: 3px solid var(--success);
            }}

            pre {{
                overflow-x: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
                margin: 0;
                font-size: 0.9em;
            }}

            code {{
                font-family: 'SF Mono', 'Menlo', 'Monaco', 'Courier New', monospace;
            }}

            .error-box {{
                background: #fef2f2;
                padding: 20px;
                border-radius: 12px;
                border-left: 4px solid var(--danger);
                margin-bottom: 25px;
            }}

            .error-box h3 {{
                color: var(--danger);
                margin-top: 0;
            }}

            /* Collapsible */
            details {{
                margin-bottom: 10px;
            }}

            details > summary {{
                list-style: none;
            }}

            details > summary::-webkit-details-marker {{
                display: none;
            }}

            /* Footer */
            .footer {{
                text-align: center;
                color: var(--gray-700);
                padding: 30px;
                font-size: 0.9em;
            }}

            /* Responsive */
            @media (max-width: 768px) {{
                .dashboard {{
                    grid-template-columns: 1fr 1fr;
                }}
                .metric-value {{
                    font-size: 1.8em;
                }}
            }}
        </style>
    </head>
    <body>
        <h1> Agent Execution Trace</h1>

        <!-- Status Banner -->
        <div class="status-banner {status_class}">
            {status_icon} {status_text}
            {f' | Final Reward: <strong>{final_reward:.4f}</strong>' if final_reward is not None else ''}
        </div>

        <!-- Dashboard Metrics -->
        <div class="dashboard">
            <div class="metric-card">
                <div class="metric-value">{stats['token_efficiency']['total_tokens']:,}</div>
                <div class="metric-label">Total Tokens</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{stats['execution']['total_steps']}</div>
                <div class="metric-label">Total Steps</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{stats['submission_analysis']['total_submissions']}</div>
                <div class="metric-label">Submissions</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{stats['submission_analysis']['best_reward'] if stats['submission_analysis']['best_reward'] is not None else 'N/A'}</div>
                <div class="metric-label">Best Reward</div>
            </div>
        </div>

        <!-- Token Breakdown -->
        <div class="section">
            <h2> Token Usage Breakdown</h2>
            <div class="token-bar">
                <div class="token-input" style="width: {stats['token_efficiency']['input_ratio']*100}%;">
                    Input: {trace.get('input_tokens_used', 0):,} ({stats['token_efficiency']['input_ratio']*100:.1f}%)
                </div>
                <div class="token-output" style="width: {stats['token_efficiency']['output_ratio']*100}%;">
                    Output: {trace.get('output_tokens_used', 0):,} ({stats['token_efficiency']['output_ratio']*100:.1f}%)
                </div>
            </div>
            <p> <strong>Efficiency:</strong> {stats['token_efficiency']['tokens_per_step']:.0f} tokens per step</p>
        </div>

        <!-- Tool Usage -->
        <div class="section">
            <h2> Tool Usage Statistics</h2>
            <div class="tool-usage-grid">
                {''.join([f'<div class="tool-card"><div class="tool-count">{count}</div><div class="tool-name">{name}</div></div>' for name, count in sorted(tool_counts.items(), key=lambda x: -x[1])])}
            </div>
        </div>

        {planets_html}

        <!-- Timeline Overview -->
        <div class="section">
            <h2> Analysis Timeline</h2>
            <div class="timeline">
                {timeline_html}
            </div>
        </div>

        {history_html}

        {'<div class="error-box"><h3> Error Occurred</h3><pre>' + safe_escape(trace.get('error_message', '')[:2000]) + '</pre></div>' if trace.get('error_message') else ''}

        <!-- Detailed Steps -->
        <div class="section">
            <h2> Detailed Execution Steps</h2>
            <details>
                <summary style="cursor: pointer; padding: 10px; background: var(--gray-100); border-radius: 8px; font-weight: 600;">
                     Click to expand all {len(trace.get('intermediate_steps', []))} steps
                </summary>
                <div style="margin-top: 20px;">
                    {''.join(steps_html)}
                </div>
            </details>
        </div>

        <div class="footer">
            <p>Generated by Stargazer Agent Execution Tracer</p>
            <p>Stop Reason: <strong>{trace.get('stop_reason', 'N/A')}</strong></p>
        </div>
    </body>
    </html>
    """

    return html
