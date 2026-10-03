"""Trace loading.

AgentGuard deliberately accepts *any* of the common trace shapes instead of
inventing yet another format.  The field-mapping strategy is borrowed from
Azure/agentops' ``trace_promotion`` (which tries a list of candidate keys),
because that is a genuinely good idea -- we just build a richer object on top
of it instead of emitting ``{input, expected}`` pairs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Step, ToolCall, Trace


class TraceError(ValueError):
    """Raised when a trace file cannot be understood."""


# Candidate key names, ordered by specificity.  Mirrors the approach used by
# Azure/agentops so that OTel / Foundry / Langfuse exports mostly "just work".
_TRACE_ID_KEYS = ("trace_id", "traceId", "operation_Id", "operationId", "id")
_STEPS_KEYS = ("steps", "turns", "messages", "conversation")
_TOOL_CALLS_KEYS = ("tool_calls", "toolCalls", "tools", "actions", "function_calls")


def _first(data: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] not in (None, "", [], {}):
            return data[key]
    return default


#: Name given to a tool call we could see but could not identify.
#:
#: Dropping such a call outright made `tool_calls=0` and a green PASS -- the
#: exact silent failure this project exists to prevent.  A trace whose tool-name
#: field is spelled differently (`fn_name`, a nested provider shape we do not
#: know) would be reported as "no tool calls, all good" rather than "I could not
#: read this".  Keeping a placeholder makes the call visible and countable;
#: no rule can match it, so it cannot cause a false positive either.
UNNAMED_TOOL = "<unnamed-tool-call>"

#: Key under ``Trace.metadata`` holding load-time warnings.  Executors and the
#: CLI read it so a partially-readable trace never reports as a clean pass.
WARNINGS_KEY = "_agentguard_warnings"


def _coerce_tool_call(raw: Any, *, turn: int) -> ToolCall | None:
    if isinstance(raw, str):
        return ToolCall(name=raw, turn=turn)
    if not isinstance(raw, dict):
        # Not a mapping at all -- genuinely unreadable, and the caller reports
        # it through the unnamed-call path so it is never silently dropped.
        return ToolCall(name=UNNAMED_TOOL, turn=turn, args={"_raw": raw})

    name = _first(raw, ("name", "tool", "tool_name", "function", "action"))
    # OpenAI style nests both name and arguments under `function`.
    function_block = raw.get("function") if isinstance(raw.get("function"), dict) else {}
    if isinstance(name, dict):  # OpenAI style: {"function": {"name": ...}}
        name = name.get("name")

    args = _first(raw, ("args", "arguments", "input", "parameters"))
    if args is None and function_block:
        args = _first(function_block, ("arguments", "args", "parameters"))
    if args is None:
        args = {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw": args}
    if not isinstance(args, dict):
        args = {"_raw": args}

    if not name:
        return ToolCall(
            name=UNNAMED_TOOL,
            args=args,
            turn=int(raw.get("turn", turn)),
            result=raw.get("result"),
        )

    return ToolCall(
        name=str(name),
        args=args,
        turn=int(raw.get("turn", turn)),
        result=raw.get("result"),
    )


def _coerce_step(raw: Any, *, index: int) -> Step:
    if isinstance(raw, str):  # plain user utterance
        return Step(turn=index, user=raw)
    if not isinstance(raw, dict):
        raise TraceError(f"step #{index} must be a mapping or a string")

    turn = int(raw.get("turn", index))

    # Chat-completions shape: {"role": "user"|"assistant"|"tool", "content": ...}
    # Without this branch the content is silently dropped, because none of the
    # flat aliases below match a role-keyed message.
    role = raw.get("role")
    if isinstance(role, str) and "content" in raw and not any(
        k in raw for k in ("user", "assistant", "input", "output")
    ):
        content = raw.get("content")
        if isinstance(content, list):  # multimodal parts
            content = " ".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        text = str(content or "")
        raw_calls = _first(raw, _TOOL_CALLS_KEYS, [])
        if isinstance(raw_calls, dict):
            raw_calls = [raw_calls]
        calls = tuple(
            call
            for item in (raw_calls or [])
            if (call := _coerce_tool_call(item, turn=turn)) is not None
        )
        if role == "assistant":
            return Step(turn=turn, assistant=text, tool_calls=calls)
        # "tool" role messages carry a result, not an utterance; keep them
        # out of user_text so intent matching is not polluted by raw payloads.
        if role == "tool":
            return Step(turn=turn, assistant="", tool_calls=calls)
        return Step(turn=turn, user=text, tool_calls=calls)

    user = _first(raw, ("user", "input", "prompt", "query", "user_message"), "")
    if isinstance(user, dict):
        user = user.get("content", "")
    assistant = _first(
        raw, ("assistant", "output", "response", "completion", "answer"), ""
    )
    if isinstance(assistant, dict):
        assistant = assistant.get("content", "")

    raw_calls = _first(raw, _TOOL_CALLS_KEYS, [])
    if isinstance(raw_calls, dict):
        raw_calls = [raw_calls]
    calls: list[ToolCall] = []
    for item in raw_calls or []:
        call = _coerce_tool_call(item, turn=turn)
        if call is not None:
            calls.append(call)

    raw_facts = _first(raw, ("facts", "recalled", "memory", "remembered"), [])
    if isinstance(raw_facts, str):
        raw_facts = [raw_facts]

    return Step(
        turn=turn,
        user=str(user or ""),
        assistant=str(assistant or ""),
        tool_calls=tuple(calls),
        facts=tuple(str(f) for f in (raw_facts or [])),
    )


def parse_trace(data: dict[str, Any], *, source: str = "unknown") -> Trace:
    if not isinstance(data, dict):
        raise TraceError("trace document must be a mapping")

    raw_steps = _first(data, _STEPS_KEYS)
    if not raw_steps:
        raise TraceError(
            f"trace has no steps; expected one of {_STEPS_KEYS}"
        )
    if not isinstance(raw_steps, (list, tuple)):
        raise TraceError("trace steps must be a list")

    steps = tuple(_coerce_step(item, index=i) for i, item in enumerate(raw_steps, 1))
    trace_id = str(_first(data, _TRACE_ID_KEYS, "trace-unknown"))

    metadata = data.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}

    unnamed = sum(
        1 for step in steps for call in step.tool_calls if call.name == UNNAMED_TOOL
    )
    if unnamed:
        # Surface it in the loaded document so every executor can report it.
        # Silence here is the failure mode: a trace whose tool names could not
        # be read looks identical to a trace that made no tool calls at all.
        metadata[WARNINGS_KEY] = [
            f"有 {unnamed} 个工具调用读不到名字："
            f"{_TOOL_CALLS_KEYS} 每一项需要 name/tool/tool_name 字段之一。"
            f"它们被计为 {UNNAMED_TOOL!r}，任何规则都匹配不到——"
            "这条 trace 没有被完整检查，别把 PASS 当成安全。"
        ]

    return Trace(
        trace_id=trace_id,
        steps=steps,
        source=source,
        metadata=metadata,
    )


def _display_path(path: Path) -> str:
    """Path as SARIF/CI wants to see it: relative to the working directory.

    Used to be ``path.name``, which threw away the directory -- so a SARIF
    result pointed at ``foo.json`` and GitHub could not resolve it to a file in
    the repository.  Falls back to the path as given when it is outside the
    working directory (e.g. a trace under ``/tmp``).
    """
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_trace(path: str | Path) -> Trace:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise TraceError(f"trace file not found: {path}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TraceError(f"{path}: invalid JSON: {exc}") from exc
    return parse_trace(data, source=_display_path(path))


def load_traces(target: str | Path) -> list[Trace]:
    path = Path(target)
    if path.is_file():
        return [load_trace(path)]
    if not path.exists():
        raise TraceError(f"trace path not found: {path}")
    traces = [load_trace(p) for p in sorted(path.rglob("*.json"))]
    if not traces:
        raise TraceError(f"no trace files found under {path}")
    return traces
