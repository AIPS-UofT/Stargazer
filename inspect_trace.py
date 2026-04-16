"""
Inspect a Stargazer trace file: print the raw messages sent to the LLM.
Usage: python inspect_trace.py <trace_json_path>
"""
import json
import sys


def truncate(s, max_len=2000):
    if len(s) > max_len:
        return s[:max_len] + f"\n... [truncated, {len(s)} chars total]"
    return s


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_trace.py <trace_json_path>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        trace = json.load(f)

    messages = trace.get("messages")

    if messages:
        # New format: full messages array saved
        print(f"Total messages in conversation: {len(messages)}\n")
        for i, msg in enumerate(messages):
            role = msg.get("role", "?")
            print("=" * 80)
            print(f"MESSAGE {i}: role={role}")
            print("=" * 80)

            # Handle different content formats
            content = msg.get("content", "")
            if isinstance(content, list):
                # Anthropic format: list of content blocks
                for block in content:
                    btype = block.get("type", "?")
                    if btype == "tool_result":
                        text = block.get("content", "")
                        tool_use_id = block.get("tool_use_id", "")
                        print(f"  [tool_result for {tool_use_id}]")
                        if "truth_planets" in str(text):
                            print("  !!! WARNING: truth_planets LEAKED !!!")
                        print(truncate(str(text)))
                    elif btype == "tool_use":
                        print(f"  [tool_use: {block.get('name', '?')}]")
                        print(truncate(json.dumps(block.get("input", {}), indent=2)))
                    elif btype == "text":
                        print(truncate(block.get("text", "")))
                    else:
                        print(truncate(json.dumps(block, indent=2)))
            elif isinstance(content, str):
                if "truth_planets" in content:
                    print("!!! WARNING: truth_planets LEAKED !!!")
                print(truncate(content))

            # Tool calls (OpenAI format)
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                func = tc.get("function", {})
                print(f"  [tool_call: {func.get('name', '?')}]")
                args = func.get("arguments", "")
                if isinstance(args, str):
                    try:
                        args = json.dumps(json.loads(args), indent=2)
                    except Exception:
                        pass
                print(truncate(str(args), 800))

            # tool_call_id (OpenAI tool response)
            if msg.get("tool_call_id"):
                print(f"  [tool_response for {msg['tool_call_id']}]")

            print()
    else:
        # Old format: reconstruct from intermediate_steps
        print("No 'messages' field in trace (old format).")
        print("Re-run the task with the updated agent to capture full messages.")
        print()
        print("Falling back to intermediate_steps reconstruction...")
        print()

        steps = trace.get("intermediate_steps", [])
        print(f"TASK INPUT:\n{'=' * 80}")
        inp = trace.get("input", "")
        print(truncate(inp, 1000))
        print()

        for i, step in enumerate(steps):
            if isinstance(step, dict):
                info = step.get("tool_info", step)
                tool_output = step.get("tool_output", "")
            elif isinstance(step, list) and len(step) >= 2:
                info = step[0]
                tool_output = step[1]
            else:
                continue

            tool = info.get("tool", "?")
            tool_input = info.get("tool_input", "")
            msg_log = info.get("message_log", None)

            print(f"{'=' * 80}")
            print(f"STEP {i}: {tool}")
            print(f"{'=' * 80}")

            if msg_log:
                print("[AGENT OUTPUT]")
                for ml in msg_log:
                    print(truncate(ml.get("content", ""), 1500))

            if tool_input and tool not in ("Assistant Text",):
                print(f"\n[TOOL CALL: {tool}]")
                s = json.dumps(tool_input, indent=2) if isinstance(tool_input, (dict, list)) else str(tool_input)
                print(truncate(s, 800))

            if tool_output and tool not in ("Assistant Text",):
                print(f"\n[ENV RESPONSE -> LLM]")
                s = str(tool_output)
                if "truth_planets" in s:
                    print("!!! WARNING: truth_planets LEAKED !!!")
                print(truncate(s))

            print()

    # Final output
    print("=" * 80)
    print("FINAL OUTPUT")
    print("=" * 80)
    print(json.dumps(trace.get("output", {}), indent=2))


if __name__ == "__main__":
    main()
