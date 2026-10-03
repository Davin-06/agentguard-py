"""Emit NeMo Guardrails tool-rail config from AgentGuard rules.

Read the NeMo docs carefully before touching this file -- the semantics are not
what they first appear to be, and getting them wrong fails *silently*.

Three facts, verified against
https://docs.nvidia.com/nemo/guardrails/configure-guardrails/guardrail-catalog/tool-calling
and ``nemoguardrails/rails/llm/config.py:437``:

1. ``rails.tool_output`` accepts **only the fixed built-in flow name**
   ``tool call validation``.  Users do **not** write custom Colang for tool
   rails.  An unknown flow name makes the config invalid for IORAILS, and NeMo
   then **silently falls back to LLMRails with every tool rail disabled** --
   the guardrail you asked for quietly stops existing.

   An earlier version of this exporter emitted ``define flow agentguard ...``
   blocks with invented Colang.  That would have disabled the very rails it
   claimed to configure.  The Colang generator has been removed.

2. NeMo's tool rail is an **allow-list over declared tools** (plus JSON-Schema
   checks on arguments).  It blocks calls to tools that were not declared.
   That maps onto AgentGuard's ``allow_tool`` -- **not** onto ``forbid_tool``,
   which has no direct equivalent here.

3. It requires the IORAILS engine, OpenAI Chat Completions wire format, and
   Colang 1.0.  None of that is expressible in this config file, so we warn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..rules import Rule

#: The only flow name ``rails.tool_output`` accepts.  Do not parameterise this.
TOOL_OUTPUT_FLOW = "tool call validation"
TOOL_INPUT_FLOW = "tool result validation"


@dataclass
class NemoExportResult:
    config: dict[str, Any]
    unsupported: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)


def _declared_tools(rules: list[Rule]) -> list[dict[str, Any]]:
    """Tools NeMo should treat as allowed, derived from ``allow_tool``."""
    names: list[str] = []
    for rule in rules:
        for tool in rule.then.allow_tool:
            if tool not in names:
                names.append(tool)
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Declared by agentguard rule set: {name}",
                # NeMo validates arguments against this schema; leaving it open
                # avoids rejecting legitimate calls we know nothing about.
                "parameters": {"type": "object", "additionalProperties": True},
            },
        }
        for name in names
    ]


def export_nemo(rules: list[Rule]) -> NemoExportResult:
    unsupported: list[str] = []
    tools = _declared_tools(rules)

    for rule in rules:
        if rule.then.forbid_tool:
            unsupported.append(
                f"{rule.name}: forbid_tool={sorted(rule.then.forbid_tool)} "
                "（NeMo 工具护栏是「已声明工具的允许列表」，没有禁止名单原语；"
                "请用 agentguard 运行时护栏或 CI 执行）"
            )
        if rule.then.allow_tool and (
            rule.when.intent_any or rule.when.turn_range is not None
        ):
            # `_declared_tools` pools every rule's allow_tool into one global
            # list, so a rule scoped to one conversation type leaks its allowed
            # tools into every other conversation type.
            conditions = []
            if rule.when.intent_any:
                conditions.append(f"intent_any={list(rule.when.intent_any)}")
            if rule.when.turn_range is not None:
                conditions.append(f"turn_range={list(rule.when.turn_range)}")
            unsupported.append(
                f"{rule.name}: when={' '.join(conditions)} "
                "（NeMo 工具护栏是**全局**允许列表，无法按对话意图切换；"
                "本规则声明的 allow_tool 会对所有会话生效）"
            )
        for spec in rule.then.idempotent:
            unsupported.append(
                f"{rule.name}: idempotent {spec.tool} on {spec.key} （NeMo 无跨调用账本）"
            )
        for spec in rule.then.memory_persistence:
            unsupported.append(
                f"{rule.name}: memory_persistence fact={spec.fact!r} "
                "（NeMo 工具护栏不涉及多轮状态）"
            )
        if rule.then.max_tool_calls is not None:
            unsupported.append(
                f"{rule.name}: max_tool_calls={rule.then.max_tool_calls} "
                "（NeMo 工具护栏不统计调用次数）"
            )

    config: dict[str, Any] = {"models": [], "rails": {}}
    if tools:
        # `rails.tool_output` only helps if NeMo knows the allowed set, so the
        # declaration and the rail are emitted together.
        config["models"] = [{"type": "main", "parameters": {"tools": tools}}]
        config["rails"]["tool_output"] = {"flows": [TOOL_OUTPUT_FLOW]}
    if unsupported:
        config["metadata"] = {
            "agentguard_unsupported": unsupported,
            "note": "这些约束无法用 NeMo 工具护栏表达，需由 agentguard 自身执行。",
        }

    requirements = [
        "需要 IORAILS 引擎：export NEMO_GUARDRAILS_IORAILS_ENGINE=1，"
        "或使用 Guardrails(config, require_iorails=True) 以避免静默回退。",
        "仅支持 OpenAI Chat Completions 线格式（openai / nim 引擎），不支持 Anthropic/Gemini。",
        "需要 Colang 1.0 配置。",
        "flows 必须严格使用内置名 'tool call validation'；自定义流程名会让工具护栏被静默禁用。",
    ]
    return NemoExportResult(config=config, unsupported=unsupported, requirements=requirements)


def render_nemo(rules: list[Rule]) -> str:
    import yaml

    result = export_nemo(rules)
    lines = [
        "# Generated by `agentguard export --to nemo`",
        "#",
        "# 重要：本文件只声明配置，不要添加自定义 Colang 流程。",
        "# rails.tool_output.flows 只接受内置名 'tool call validation'；",
        "# 写错流程名会让 NeMo 静默回退并禁用全部工具护栏。",
        "#",
        "# 前置条件：",
    ]
    lines.extend(f"#   - {item}" for item in result.requirements)
    if result.unsupported:
        lines.append("#")
        lines.append("# 以下约束未能导出（NeMo 无对应原语）：")
        lines.extend(f"#   - {item}" for item in result.unsupported)
    body = yaml.safe_dump(result.config, allow_unicode=True, sort_keys=False)
    return "\n".join(lines) + "\n" + body
