"""Chinese renderings for violation kinds and runtime denial codes.

The canonical ``Violation.message`` / ``GateDecision.reason`` strings stay
English on purpose.  They end up in production logs, CI JSON payloads and
``ToolBlocked`` tracebacks -- places where a stable ASCII identifier is easier
to grep and easier to paste to a teammate.  Only the *human-facing* rendering
(terminal output and the markdown report) is localised here, keyed by the
machine-readable ``kind`` / ``code``.

An unknown kind falls back to the canonical message rather than printing
nothing: a report that silently drops a violation is worse than a report with
one English line in it.
"""

from __future__ import annotations

from typing import Any


def violation_zh(
    *,
    kind: str,
    message: str,
    detail: dict[str, Any] | None = None,
    turn: int | None = None,
    tool: str | None = None,
) -> str:
    """Render a CI violation in Chinese, falling back to the canonical message.

    The renderers already print the turn in a dedicated column, so the messages
    below deliberately do not repeat it -- except for ``duplicate_transaction``,
    where the *pair* of turns is the whole point.
    """
    d = detail or {}

    if kind == "forbidden_tool":
        return f"调用了被规则禁止的工具 {tool!r}"

    if kind == "tool_not_allowed":
        return (
            f"工具 {tool!r} 不在允许列表 {d.get('allow_tool')} 内"
            "（allow_tool 是「除名单外全拒」）"
        )

    if kind == "missing_required_tool":
        return f"必需的工具 {tool!r} 从未被调用"

    if kind == "too_many_tool_calls":
        return f"工具调用共 {d.get('observed')} 次，超过上限 {d.get('limit')} 次"

    if kind == "idempotency_key_unresolved":
        return (
            f"无法解析幂等键 {d.get('key_template')!r}"
            f"（工具 {tool!r}）—— 该次调用未做重复检查"
        )

    if kind == "duplicate_transaction":
        label = str(d.get("key_template", "")).strip("${}") or d.get("key_template")
        return (
            f"重复事务：{tool!r} 已用同一 {label}={d.get('key_value')!r} 执行过"
            f"（第 {d.get('first_turn')} 轮与第 {turn} 轮）"
        )

    if kind == "memory_turn_missing":
        return f"该轮不在 trace 中，无法验证是否记住了 {d.get('fact')!r}"

    if kind == "memory_not_recalled":
        return f"没有遵守 {d.get('fact')!r}（该事实在第 {d.get('inject_at')} 轮注入）"

    return message


def decision_zh(
    *,
    code: str,
    reason: str,
    detail: dict[str, Any] | None = None,
    rule: str | None = None,
) -> str:
    """Render a runtime allow/deny decision in Chinese."""
    d = detail or {}

    if code == "allowed":
        return "没有规则提出异议"

    if code == "forbidden_tool":
        return f"工具被规则 {rule!r} 禁止 —— 业务函数不会被执行"

    if code == "tool_not_allowed":
        return (
            f"工具不在规则 {rule!r} 的允许列表 {d.get('allow_tool')} 内"
            "（allow_tool 是「除名单外全拒」）"
        )

    if code == "duplicate_transaction":
        return (
            f"重复事务：该 {d.get('key_name', 'key')}={d.get('key')!r} "
            f"已经执行过一次（规则 {rule!r}）"
        )

    return reason
