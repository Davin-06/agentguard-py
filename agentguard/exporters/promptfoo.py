"""Emit promptfoo test config from AgentGuard rules.

Verified against promptfoo's source that ``not-`` prefixing genuinely inverts a
trajectory assertion (``src/assertions/trajectory.ts:49 applyInverse()``), so a
forbidden tool maps cleanly onto ``not-trajectory:tool-used``.

**What this module deliberately does not do:** invent JavaScript assertions for
capabilities promptfoo has no primitive for.  An earlier version emitted
``javascript`` assertions reading ``context.vars.__trajectory`` -- a variable
promptfoo never provides (the real field is ``context.trace``, whose span shape
is too involved to parse reliably from generated code).  Those assertions would
have failed at runtime while looking correct in the YAML.  Unsupported
constraints are now reported instead of faked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..rules import Rule


@dataclass
class ExportResult:
    """A generated promptfoo config plus an honest account of the gaps."""

    config: dict[str, Any]
    unsupported: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"导出 {len(self.config.get('tests', []))} 个 test case"]
        if self.unsupported:
            lines.append("以下约束 promptfoo 无对应原语，未生成断言（需自行处理）：")
            lines.extend(f"  - {item}" for item in self.unsupported)
        return "\n".join(lines)


def _assertions_for(rule: Rule) -> tuple[list[dict[str, Any]], list[str]]:
    """Map a rule onto promptfoo assertions.

    Returns the assertions plus the constraints that could not be expressed.
    ``allow_tool`` and ``idempotent`` are intentionally absent -- promptfoo
    offers no allow-list primitive and no cross-call ledger.
    """
    asserts: list[dict[str, Any]] = []
    unsupported: list[str] = []

    for tool in rule.then.forbid_tool:
        asserts.append(
            {
                "type": "not-trajectory:tool-used",
                "value": tool,
                "metric": f"agentguard/{rule.name}/forbid-{tool}",
            }
        )

    for tool in rule.then.require_tool:
        asserts.append(
            {
                "type": "trajectory:tool-used",
                "value": tool,
                "metric": f"agentguard/{rule.name}/require-{tool}",
            }
        )

    if rule.then.max_tool_calls is not None:
        asserts.append(
            {
                "type": "trajectory:step-count",
                "value": {"type": "tool", "max": rule.then.max_tool_calls},
                "metric": f"agentguard/{rule.name}/call-budget",
            }
        )

    # The `when` block gates the whole rule, and promptfoo has no conditional
    # assertion primitive.  Emitting the assertions anyway silently *widens*
    # them: a rule that only applied to logistics conversations becomes an
    # unconditional ban, so legitimate refund tests start failing.  Say so.
    if asserts and (rule.when.intent_any or rule.when.turn_range is not None):
        conditions = []
        if rule.when.intent_any:
            conditions.append(f"intent_any={list(rule.when.intent_any)}")
        if rule.when.turn_range is not None:
            conditions.append(f"turn_range={list(rule.when.turn_range)}")
        unsupported.append(
            f"{rule.name}: when={' '.join(conditions)} "
            "（promptfoo 无「仅当…时」的条件断言原语；"
            "下面导出的断言会在**所有**场景生效，比原规则更严格，可能误报）"
        )

    if rule.then.allow_tool:
        unsupported.append(
            f"{rule.name}: allow_tool={sorted(rule.then.allow_tool)} "
            "（promptfoo 无 allow-list 原语；请用 agentguard 自身在 CI 执行）"
        )

    for spec in rule.then.idempotent:
        unsupported.append(
            f"{rule.name}: idempotent {spec.tool} on {spec.key} "
            "（promptfoo 无跨调用账本；请用 agentguard 自身在 CI 执行）"
        )

    for spec in rule.then.memory_persistence:
        unsupported.append(
            f"{rule.name}: memory_persistence fact={spec.fact!r} "
            "（promptfoo 只有整段 conversation-relevance，无 per-turn 断言）"
        )

    return asserts, unsupported


def export_promptfoo(rules: list[Rule]) -> ExportResult:
    """Build a promptfoo config mapping each rule to one test case.

    The ``providers`` block is a **placeholder** and is emitted as such: how you
    replay a trace through promptfoo depends entirely on your agent, and
    inventing a provider config here would only look plausible.
    """
    tests: list[dict[str, Any]] = []
    unsupported: list[str] = []

    for rule in rules:
        asserts, gaps = _assertions_for(rule)
        unsupported.extend(gaps)
        if not asserts:
            # Nothing expressible -- do not emit an empty test case that would
            # silently pass and give a false sense of coverage.
            continue
        entry: dict[str, Any] = {
            "description": f"[agentguard] {rule.name}",
            "vars": {},
            "assert": asserts,
            "metadata": {"agentguard_rule": rule.name, "severity": rule.severity},
        }
        # Deliberately *not* exporting `when.intent_any` as a variable.
        #
        # An earlier version wrote it to `vars.__agentguard_intent_any`, but no
        # assertion ever read it -- promptfoo has no conditional-assertion
        # primitive to consume it with.  So the config carried a variable that
        # looked exactly like the gate surviving the export, while the gate had
        # in fact been dropped (the assertion fires unconditionally).  The
        # orphaned variable actively argues against the warning printed right
        # above it, and a user who trusts the YAML over the stderr gets a config
        # that fails on legitimate refund tests.
        #
        # The condition is recorded in `metadata.agentguard_when` instead:
        # still available to tooling, but unmistakably documentation of an
        # unported constraint rather than a live binding.
        if rule.when.intent_any or rule.when.turn_range is not None:
            when: dict[str, Any] = {"ported": False}
            if rule.when.intent_any:
                when["intent_any"] = list(rule.when.intent_any)
            if rule.when.turn_range is not None:
                when["turn_range"] = list(rule.when.turn_range)
            entry["metadata"]["agentguard_when"] = when
        tests.append(entry)

    config: dict[str, Any] = {
        "description": "Generated by `agentguard export --to promptfoo`",
        "prompts": ["{{prompt}}"],
        "providers": [{"id": "agentguard-replay", "config": {"traces": "traces/"}}],
        "tests": tests,
    }
    if unsupported:
        config["metadata"] = {
            "agentguard_unsupported": unsupported,
            "note": "这些约束无法用 promptfoo 原语表达，需由 agentguard 自身执行。",
        }
    return ExportResult(config=config, unsupported=unsupported)


def render_promptfoo(rules: list[Rule]) -> str:
    import yaml

    result = export_promptfoo(rules)
    header = "# Generated by `agentguard export --to promptfoo`\n"
    if result.unsupported:
        header += "#\n# 注意：以下约束未能导出（promptfoo 无对应原语）\n"
        header += "".join(f"#   - {item}\n" for item in result.unsupported)
    body = yaml.safe_dump(result.config, allow_unicode=True, sort_keys=False)
    return f"{header}{body}"
