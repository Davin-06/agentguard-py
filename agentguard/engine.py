"""The evaluation engine.

Design rule that everything else follows from:

    **Only deterministic checks may produce a violation.**

No LLM judge runs in the gate path.  That is a direct answer to the single
biggest reason teams stop trusting eval tools (flaky judges -> false alarms ->
the gate gets disabled).  Semantic checks can be layered on later, but they
must never be the thing that blocks a deploy.
"""

from __future__ import annotations

import re
from typing import Any

from .models import RuleResult, Severity, Trace, Violation
from .rules import Rule

_PLACEHOLDER = re.compile(r"\$\{([^}]+)\}")


class _MissingType:
    """Sentinel type for an unresolvable placeholder."""

    _instance: "_MissingType | None" = None

    def __new__(cls) -> "_MissingType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unresolved>"

    def __bool__(self) -> bool:
        return False


#: Singleton sentinel returned by :func:`render_template` when a placeholder
#: cannot be resolved.  Use :func:`is_missing` rather than ``isinstance`` so the
#: check works no matter how the value travelled.
MISSING = _MissingType()


def is_missing(value: object) -> bool:
    """True when ``value`` is the unresolved-placeholder sentinel."""
    return isinstance(value, _MissingType)


def render_template(template: str, call_args: dict[str, Any]) -> Any:
    """Render ``"${order_id}"`` against tool-call args.

    Returns the raw value (not a string) when the whole template is a single
    placeholder, so ``key: "${order.id}"`` can yield an int.
    """
    matches = list(_PLACEHOLDER.finditer(template))
    if not matches:
        return template

    def _lookup(expr: str) -> Any:
        current: Any = call_args
        for part in expr.strip().split("."):
            if not isinstance(current, dict) or part not in current:
                return MISSING
            current = current[part]
        return current

    if len(matches) == 1 and matches[0].group(0) == template:
        return _lookup(matches[0].group(1))

    def _sub(match: re.Match[str]) -> str:
        value = _lookup(match.group(1))
        return "" if is_missing(value) else str(value)

    return _PLACEHOLDER.sub(_sub, template)


def rule_applies(rule: Rule, trace: Trace) -> bool:
    """Whether the rule's ``when`` block activates for this trace."""
    when = rule.when
    if when.always:
        return True

    if when.intent_any:
        haystack = trace.user_text().lower()
        if not any(kw.lower() in haystack for kw in when.intent_any):
            return False

    if when.turn_range is not None:
        lo, hi = when.turn_range
        if not any(lo <= step.turn <= hi for step in trace.steps):
            return False

    return True


def evaluate(rule: Rule, trace: Trace) -> RuleResult:
    """Evaluate one rule against one trace."""
    if not rule_applies(rule, trace):
        return RuleResult(rule=rule.name, applied=False)

    violations: list[Violation] = []
    violations.extend(_check_forbid(rule, trace))
    violations.extend(_check_allow(rule, trace))
    violations.extend(_check_require(rule, trace))
    violations.extend(_check_max_calls(rule, trace))
    violations.extend(_check_idempotency(rule, trace))
    violations.extend(_check_memory(rule, trace))
    return RuleResult(
        rule=rule.name, applied=True, violations=tuple(_dedupe(violations))
    )


def _dedupe(violations: list[Violation]) -> list[Violation]:
    """Collapse violations that describe the very same offending event.

    A rule saying ``forbid_tool: [refund_order]`` plus
    ``allow_tool: [query_order]`` is stating one fact twice -- the forbidden
    tool cannot be inside the allow-list.  Reporting it as two violations
    inflated counts, nudged the ``--baseline`` comparison, and produced two
    SARIF alerts for one bad call.

    Only ``forbidden_tool`` / ``tool_not_allowed`` collapse, and only when they
    point at the same call: those two are logical complements of each other,
    so for a given (tool, turn) at most one of them can be new information.
    Everything else keeps its own entry -- two duplicate transactions at
    different turns are genuinely two findings.
    """
    complementary = {"forbidden_tool", "tool_not_allowed"}
    out: list[Violation] = []
    seen: set[tuple[str, str, int | None]] = set()

    for violation in violations:
        if violation.kind in complementary and violation.tool is not None:
            identity = (violation.tool, "tool_call", violation.turn)
            if identity in seen:
                continue
            seen.add(identity)
        out.append(violation)
    return out


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #
def _check_forbid(rule: Rule, trace: Trace) -> list[Violation]:
    out: list[Violation] = []
    forbidden = set(rule.then.forbid_tool)
    if not forbidden:
        return out
    for call in trace.tool_calls():
        if call.name in forbidden:
            out.append(
                Violation(
                    rule=rule.name,
                    kind="forbidden_tool",
                    severity=rule.severity,
                    message=(
                        f"Tool {call.name!r} must not be called, "
                        f"but it was invoked at turn {call.turn}"
                    ),
                    turn=call.turn,
                    tool=call.name,
                    detail={"args": call.args},
                )
            )
    return out


def _check_allow(rule: Rule, trace: Trace) -> list[Violation]:
    out: list[Violation] = []
    allowed = set(rule.then.allow_tool)
    if not allowed:
        return out
    for call in trace.tool_calls():
        if call.name not in allowed:
            out.append(
                Violation(
                    rule=rule.name,
                    kind="tool_not_allowed",
                    severity=rule.severity,
                    message=(
                        f"Tool {call.name!r} is outside the allow-list "
                        f"{sorted(allowed)} (turn {call.turn})"
                    ),
                    turn=call.turn,
                    tool=call.name,
                    detail={"allow_tool": sorted(allowed)},
                )
            )
    return out


def _check_require(rule: Rule, trace: Trace) -> list[Violation]:
    out: list[Violation] = []
    required = set(rule.then.require_tool)
    if not required:
        return out
    seen = set(trace.tools())
    for tool in sorted(required - seen):
        out.append(
            Violation(
                rule=rule.name,
                kind="missing_required_tool",
                severity=rule.severity,
                message=f"Required tool {tool!r} was never called",
                tool=tool,
            )
        )
    return out


def _check_max_calls(rule: Rule, trace: Trace) -> list[Violation]:
    limit = rule.then.max_tool_calls
    if limit is None:
        return []
    total = len(trace.tool_calls())
    if total <= limit:
        return []
    return [
        Violation(
            rule=rule.name,
            kind="too_many_tool_calls",
            severity=rule.severity,
            message=f"Tool calls {total} exceed the limit of {limit}",
            detail={"observed": total, "limit": limit},
        )
    ]


def _check_idempotency(rule: Rule, trace: Trace) -> list[Violation]:
    """Catch the "refund executed twice" class of incident.

    This is the check that no general-purpose eval framework ships, because it
    needs business-level knowledge of *which argument identifies the
    transaction*.
    """
    out: list[Violation] = []
    for spec in rule.then.idempotent:
        seen: dict[str, int] = {}
        for call in trace.tool_calls():
            if call.name != spec.tool:
                continue
            key = render_template(spec.key, call.args)
            if is_missing(key):
                out.append(
                    Violation(
                        rule=rule.name,
                        kind="idempotency_key_unresolved",
                        severity="warning",
                        message=(
                            f"Could not resolve idempotency key {spec.key!r} "
                            f"for tool {spec.tool!r} at turn {call.turn}"
                        ),
                        turn=call.turn,
                        tool=call.name,
                        detail={"key_template": spec.key, "args": call.args},
                    )
                )
                continue
            token = repr(key)
            if token in seen:
                label = spec.key.strip("${}") or spec.key
                out.append(
                    Violation(
                        rule=rule.name,
                        kind="duplicate_transaction",
                        severity=rule.severity,
                        message=(
                            f"Tool {spec.tool!r} executed more than once for the same "
                            f"{label}={key!r} (turns {seen[token]} and {call.turn})"
                        ),
                        turn=call.turn,
                        tool=call.name,
                        detail={
                            "key_template": spec.key,
                            "key_value": key,
                            "first_turn": seen[token],
                        },
                    )
                )
            else:
                seen[token] = call.turn
    return out


def _check_memory(rule: Rule, trace: Trace) -> list[Violation]:
    """Per-turn state persistence -- the gap every other tool leaves open.

    promptfoo only offers whole-conversation ``conversation-relevance``; there
    is no per-turn "must still remember fact F at turn N" primitive.
    """
    out: list[Violation] = []
    for spec in rule.then.memory_persistence:
        for turn in spec.must_recall_at:
            step = trace.turn(turn)
            if step is None:
                out.append(
                    Violation(
                        rule=rule.name,
                        kind="memory_turn_missing",
                        severity="warning",
                        message=f"Turn {turn} is missing from the trace; cannot verify recall",
                        turn=turn,
                        detail={"fact": spec.fact},
                    )
                )
                continue
            if not _fact_recalled(spec.fact, step.facts, step.assistant):
                out.append(
                    Violation(
                        rule=rule.name,
                        kind="memory_not_recalled",
                        severity=rule.severity,
                        message=(
                            f"Fact {spec.fact!r} was not honoured at turn {turn} "
                            f"(injected at turn {spec.inject_at})"
                        ),
                        turn=turn,
                        detail={"fact": spec.fact, "inject_at": spec.inject_at},
                    )
                )
    return out


def _fact_recalled(fact: str, facts: tuple[str, ...], assistant: str) -> bool:
    """Deterministic fact recall check.

    Looks at the explicit ``facts`` list first, then falls back to a literal
    substring test on the assistant text.  Deliberately *not* a semantic
    comparison -- see the module docstring.
    """
    needle = fact.strip().lower()
    if any(needle == f.strip().lower() for f in facts):
        return True
    if any(needle in f.strip().lower() for f in facts):
        return True
    return needle in assistant.lower()


def evaluate_all(rules: list[Rule], trace: Trace) -> list[RuleResult]:
    return [evaluate(rule, trace) for rule in rules]


def worst_severity(violations: tuple[Violation, ...] | list[Violation]) -> Severity | None:
    from .models import SEVERITY_ORDER

    if not violations:
        return None
    return max((v.severity for v in violations), key=lambda s: SEVERITY_ORDER.get(s, 0))
