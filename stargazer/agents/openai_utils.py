from __future__ import annotations

"""Utilities for routing OpenAI calls between chat.completions and the Responses API."""

import json
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Union


def _normalize_function_call_id(call_id: Optional[str]) -> Optional[str]:
    """
    Responses API requires function call ids to start with ``fc_``.
    Older chat-style ids often start with ``call_``; coerce them to the
    required prefix so function_call and function_call_output items line up.
    """
    if not call_id:
        return None
    if call_id.startswith("fc_"):
        return call_id
    if call_id.startswith("call_"):
        call_id = call_id[len("call_") :]
    return f"fc_{call_id}"


def _as_dict(obj: Any) -> Dict[str, Any]:
    """Best-effort conversion of OpenAI SDK models to dictionaries."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    to_dict = getattr(obj, "model_dump", None)
    if callable(to_dict):
        return to_dict()
    attrs: Dict[str, Any] = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        try:
            value = getattr(obj, key)
        except AttributeError:
            continue
        if callable(value):
            continue
        attrs[key] = value
    return attrs


def _is_gpt5_series_model(model: Optional[str]) -> bool:
    if not model:
        return False
    lower = model.lower()
    return lower.startswith("gpt-5")


def should_use_responses_api(model: Optional[str]) -> bool:
    """Whether the model requires OpenAI's newer Responses API (only gpt-5 series)."""
    return _is_gpt5_series_model(model)


def _coerce_text_from_content(content: Union[str, Iterable[Any], None]) -> str:
    """Extract plain text from the Responses API content blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    text_parts: List[str] = []
    for part in content:
        if isinstance(part, str):
            text_parts.append(part)
            continue
        part_dict = _as_dict(part)
        part_type = part_dict.get("type")
        text_value = part_dict.get("text") or part_dict.get("content")
        if part_type in {"output_text", "input_text", "text"} and text_value:
            text_parts.append(text_value)
    return "\n".join(p.strip() for p in text_parts if p.strip())


def _build_usage_adapter(response_usage: Any) -> SimpleNamespace:
    usage_dict = _as_dict(response_usage)
    prompt_tokens = usage_dict.get("prompt_tokens", usage_dict.get("input_tokens", 0))
    completion_tokens = usage_dict.get(
        "completion_tokens", usage_dict.get("output_tokens", 0)
    )
    return SimpleNamespace(
        prompt_tokens=prompt_tokens or 0,
        completion_tokens=completion_tokens or 0,
    )


def _extract_message_from_response(response: Any) -> SimpleNamespace:
    response_dict = _as_dict(response)
    output_items = response_dict.get("output") or []

    # 新格式：直接从 output 中提取工具调用和文本
    text_parts: List[str] = []
    function_calls: List[Dict[str, Any]] = []
    message_dict = None

    for item in output_items:
        item_dict = _as_dict(item)
        item_type = item_dict.get("type") or item_dict.get("object")

        # 旧格式：查找 message 类型
        if item_type == "message":
            message_dict = item_dict.get("message") or item_dict
            break

        # 新格式：处理 function_call 类型
        elif item_type == "function_call":
            call_id = _normalize_function_call_id(
                item_dict.get("call_id") or item_dict.get("id")
            )
            function_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item_dict.get("name"),
                        "arguments": item_dict.get("arguments"),
                    },
                }
            )

        # 新格式：提取文本内容
        elif item_type == "output_text" or item_type == "text":
            text_value = item_dict.get("text") or item_dict.get("content")
            if text_value:
                text_parts.append(text_value)

    # 如果找到了旧格式的 message，使用旧逻辑
    if message_dict is not None:
        text_content = _coerce_text_from_content(message_dict.get("content"))
        tool_calls = _normalize_tool_calls(message_dict.get("tool_calls"))
        return SimpleNamespace(
            content=text_content,
            tool_calls=tool_calls,
            role=message_dict.get("role", "assistant"),
        )

    # 检查 response 顶层是否有 message（兼容性）
    if "message" in response_dict:
        message_dict = response_dict["message"]
        text_content = _coerce_text_from_content(message_dict.get("content"))
        tool_calls = _normalize_tool_calls(message_dict.get("tool_calls"))
        return SimpleNamespace(
            content=text_content,
            tool_calls=tool_calls,
            role=message_dict.get("role", "assistant"),
        )

    # 新格式：返回收集到的内容和工具调用
    text_content = "\n".join(p.strip() for p in text_parts if p.strip())
    tool_calls = _normalize_tool_calls(function_calls) if function_calls else None

    return SimpleNamespace(
        content=text_content,
        tool_calls=tool_calls,
        role="assistant",
    )


def _normalize_tool_calls(tool_calls: Any) -> Optional[List[SimpleNamespace]]:
    normalized: List[SimpleNamespace] = []
    for call in tool_calls or []:
        call_dict = _as_dict(call)
        function_dict = _as_dict(call_dict.get("function"))
        arguments = function_dict.get("arguments")
        if arguments is not None and not isinstance(arguments, str):
            try:
                arguments = json.dumps(arguments)
            except TypeError:
                arguments = str(arguments)
        normalized.append(
            SimpleNamespace(
                id=_normalize_function_call_id(call_dict.get("id")),
                type=call_dict.get("type", "function"),
                function=SimpleNamespace(
                    name=function_dict.get("name"),
                    arguments=arguments,
                ),
            )
        )
    return normalized or None


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #

def _convert_tools_for_responses(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Convert Chat Completions tool schema to Responses API format."""
    if not tools:
        return tools
    converted: List[Dict[str, Any]] = []
    for tool in tools:
        tool_type = tool.get("type")
        if tool_type != "function":
            converted.append(tool)
            continue
        function_block = tool.get("function", {})
        converted.append(
            {
                "type": "function",
                "name": function_block.get("name"),
                "description": function_block.get("description"),
                "parameters": function_block.get("parameters", {}),
            }
        )
    return converted


def _convert_messages_for_responses(
    messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Translate Chat Completions style messages to Responses API input blocks."""

    def _content_to_blocks(content: Any, role: Optional[str]) -> List[Dict[str, Any]]:
        if content is None:
            return []
        if isinstance(content, list):
            return content
        # Default to plain text block for strings or other scalar types
        block_type = "output_text" if role == "assistant" else "input_text"
        return [{"type": block_type, "text": str(content)}]

    converted: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        # Tool responses are represented as function_call_output items in the Responses API.
        if role == "tool":
            call_id = _normalize_function_call_id(
                msg.get("tool_call_id") or msg.get("id")
            )
            converted.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": _coerce_text_from_content(msg.get("content")),
                }
            )
            continue

        content_blocks = _content_to_blocks(msg.get("content"), role)
        converted_msg: Dict[str, Any] = {
            "role": role,
            "content": content_blocks,
            "type": "message",
        }
        converted.append(converted_msg)

        # Function calls should be separate items, not embedded inside message content.
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                function_block = tc.get("function", {})
                call_id = _normalize_function_call_id(
                    tc.get("id") or tc.get("call_id")
                )
                converted.append(
                    {
                        "type": "function_call",
                        "id": call_id,
                        "call_id": call_id,
                        "name": function_block.get("name"),
                        "arguments": function_block.get("arguments"),
                    }
                )

    return converted


def create_openai_chat_completion(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    use_responses_api: Optional[bool] = None,
    stream: bool = False,
    on_text_delta: Optional[Any] = None,
) -> Any:
    """
    Call OpenAI's chat API, automatically routing to the Responses API for models that require it.

    Returns an object with `.choices[0].message` and `.usage` attributes analogous to the Chat Completions API.
    """
    use_responses = (
        should_use_responses_api(model)
        if use_responses_api is None
        else bool(use_responses_api)
    )
    if use_responses:
        response_kwargs: Dict[str, Any] = {
            "model": model,
            "input": _convert_messages_for_responses(messages),
        }
        if tools:
            response_kwargs["tools"] = _convert_tools_for_responses(tools)
        # gpt-5* Responses API currently rejects temperature; drop it to avoid errors.
        if max_tokens is not None:
            response_kwargs["max_output_tokens"] = max_tokens
        if reasoning_effort:
            response_kwargs["reasoning"] = {"effort": reasoning_effort}
        response = client.responses.create(**response_kwargs)
        assistant_message = _extract_message_from_response(response)
        usage_adapter = _build_usage_adapter(getattr(response, "usage", None))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=assistant_message)],
            usage=usage_adapter,
        )

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if tools and not use_responses:
        payload["tools"] = tools
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if not stream:
        return client.chat.completions.create(**payload)

    stream_response = client.chat.completions.create(
        **payload,
        stream=True,
        stream_options={"include_usage": True},
    )
    content_parts: List[str] = []
    tool_calls_by_index: Dict[int, Dict[str, Any]] = {}
    usage_adapter = SimpleNamespace(prompt_tokens=0, completion_tokens=0)

    for chunk in stream_response:
        chunk_dict = _as_dict(chunk)

        usage_dict = _as_dict(chunk_dict.get("usage"))
        if usage_dict:
            usage_adapter = _build_usage_adapter(usage_dict)

        choices = chunk_dict.get("choices") or []
        if not choices:
            continue

        delta = _as_dict(choices[0].get("delta"))
        if not delta:
            continue

        content_delta = delta.get("content")
        if isinstance(content_delta, str):
            if content_delta:
                content_parts.append(content_delta)
                if callable(on_text_delta):
                    on_text_delta(content_delta)
        elif isinstance(content_delta, list):
            for block in content_delta:
                block_dict = _as_dict(block)
                text = block_dict.get("text")
                if text:
                    content_parts.append(text)
                    if callable(on_text_delta):
                        on_text_delta(text)

        for tc_delta in delta.get("tool_calls") or []:
            tc = _as_dict(tc_delta)
            idx = int(tc.get("index", 0))
            current = tool_calls_by_index.setdefault(
                idx,
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": None, "arguments": ""},
                },
            )
            if tc.get("id"):
                current["id"] = tc["id"]
            function_delta = _as_dict(tc.get("function"))
            if function_delta.get("name"):
                current["function"]["name"] = function_delta["name"]
            if function_delta.get("arguments"):
                current["function"]["arguments"] += function_delta["arguments"]

    ordered_tool_calls = [
        tool_calls_by_index[i] for i in sorted(tool_calls_by_index.keys())
    ]
    assistant_message = SimpleNamespace(
        role="assistant",
        content="".join(content_parts),
        tool_calls=_normalize_tool_calls(ordered_tool_calls) if ordered_tool_calls else None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=assistant_message)],
        usage=usage_adapter,
    )
