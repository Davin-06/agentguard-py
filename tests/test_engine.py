"""Engine tests -- every assertion kind, both positive and negative."""

from __future__ import annotations

import pytest

from agentguard.engine import evaluate, render_template, rule_applies
from agentguard.models import Step, ToolCall, Trace
from agentguard.rules import (
    IdempotencyRule,
    MemoryRule,
    Rule,
    Then,
    When,
    Where,
    parse_rule,
)


def _rule(**then_kwargs) -> Rule:
    return Rule(name="t", then=Then(**then_kwargs))


# --------------------------------------------------------------------------- #
# render_template
# --------------------------------------------------------------------------- #
def test_render_template_single_placeholder_returns_raw_value():
    assert render_template("${order_id}", {"order_id": 42}) == 42


def test_render_template_dotted_path():
    assert render_template("${order.id}", {"order": {"id": "X"}}) == "X"


def test_render_template_missing_key_is_sentinel_not_crash():
    from agentguard.engine import is_missing

    assert is_missing(render_template("${nope}", {}))


def test_render_template_interpolates_inside_string():
    assert render_template("k-${id}-v", {"id": 7}) == "k-7-v"


# --------------------------------------------------------------------------- #
# when
# --------------------------------------------------------------------------- #
def test_rule_without_when_always_applies(refund_trace):
    assert rule_applies(_rule(forbid_tool=("x",)), refund_trace)


def test_intent_gate_blocks_non_matching_trace():
    rule = Rule(
        name="r",
        then=Then(forbid_tool=("refund_order",)),
        when=When(intent_any=("完全不相关的词",)),
    )
    trace = Trace(
        trace_id="t", steps=(Step(turn=1, user="你好"),)
    )
    assert not rule_applies(rule, trace)


def test_turn_range_gate():
    rule = Rule(
        name="r",
        then=Then(forbid_tool=("x",)),
        when=When(turn_range=(10, 20)),
    )
    short = Trace(trace_id="t", steps=(Step(turn=1, user="hi"),))
    assert not rule_applies(rule, short)


# --------------------------------------------------------------------------- #
# forbid / allow / require
# --------------------------------------------------------------------------- #
def test_forbid_tool_catches_every_occurrence(refund_trace):
    result = evaluate(_rule(forbid_tool=("refund_order",)), refund_trace)
    assert not result.passed
    assert len(result.violations) == 2
    assert {v.kind for v in result.violations} == {"forbidden_tool"}


def test_allow_list_flags_outsiders(refund_trace):
    result = evaluate(
        _rule(allow_tool=("query_order", "get_logistics")), refund_trace
    )
    assert not result.passed
    assert all(v.kind == "tool_not_allowed" for v in result.violations)
    assert all(v.tool == "refund_order" for v in result.violations)


def test_require_tool_reports_missing():
    rule = _rule(require_tool=("must_call_me",))
    trace = Trace(trace_id="t", steps=(Step(turn=1, user="hi"),))
    result = evaluate(rule, trace)
    assert not result.passed
    assert result.violations[0].kind == "missing_required_tool"


def test_max_tool_calls():
    rule = _rule(max_tool_calls=1)
    trace = Trace(
        trace_id="t",
        steps=(
            Step(turn=1, user="a", tool_calls=(ToolCall("x"), ToolCall("y"))),
        ),
    )
    result = evaluate(rule, trace)
    assert not result.passed
    assert result.violations[0].kind == "too_many_tool_calls"


# --------------------------------------------------------------------------- #
# idempotency -- the check no general eval framework ships
# --------------------------------------------------------------------------- #
def test_idempotency_detects_duplicate_transaction(refund_trace):
    rule = _rule(idempotent=(IdempotencyRule("refund_order", "${order_id}"),))
    result = evaluate(rule, refund_trace)
    assert not result.passed
    kinds = {v.kind for v in result.violations}
    assert "duplicate_transaction" in kinds
    detail = result.violations[0].detail
    assert detail["key_value"] == "A1001"
    assert detail["first_turn"] == 3


def test_idempotency_allows_distinct_keys():
    rule = _rule(idempotent=(IdempotencyRule("refund_order", "${order_id}"),))
    trace = Trace(
        trace_id="t",
        steps=(
            Step(turn=1, user="a", tool_calls=(ToolCall("refund_order", {"order_id": "A"}),)),
            Step(turn=2, user="b", tool_calls=(ToolCall("refund_order", {"order_id": "B"}),)),
        ),
    )
    assert evaluate(rule, trace).passed


def test_idempotency_unresolved_key_warns_not_fails():
    rule = _rule(idempotent=(IdempotencyRule("refund_order", "${missing}"),))
    trace = Trace(
        trace_id="t",
        steps=(Step(turn=1, user="a", tool_calls=(ToolCall("refund_order", {}),)),),
    )
    result = evaluate(rule, trace)
    assert result.violations[0].kind == "idempotency_key_unresolved"
    assert result.violations[0].severity == "warning"


# --------------------------------------------------------------------------- #
# memory persistence -- the per-turn gap
# --------------------------------------------------------------------------- #
def test_memory_not_recalled_is_flagged():
    rule = _rule(
        memory_persistence=(
            MemoryRule(fact="预算上限 5000", inject_at=2, must_recall_at=(5, 20)),
        )
    )
    trace = Trace(
        trace_id="t",
        steps=(
            Step(turn=5, user="a", assistant="预算内推荐", facts=("预算上限 5000",)),
            Step(turn=20, user="下单", assistant="已选旗舰款 8999"),
        ),
    )
    result = evaluate(rule, trace)
    assert not result.passed
    assert len(result.violations) == 1
    assert result.violations[0].turn == 20
    assert result.violations[0].kind == "memory_not_recalled"


def test_memory_recall_passes_when_fact_present_at_every_checkpoint():
    rule = _rule(
        memory_persistence=(
            MemoryRule(fact="预算上限 5000", must_recall_at=(5, 10)),
        )
    )
    trace = Trace(
        trace_id="t",
        steps=(
            Step(turn=5, user="a", facts=("预算上限 5000",)),
            Step(turn=10, user="b", facts=("预算上限 5000",)),
        ),
    )
    assert evaluate(rule, trace).passed


def test_memory_missing_turn_reports_clearly():
    rule = _rule(
        memory_persistence=(MemoryRule(fact="F", must_recall_at=(99,)),)
    )
    trace = Trace(trace_id="t", steps=(Step(turn=1, user="a"),))
    result = evaluate(rule, trace)
    assert result.violations[0].kind == "memory_turn_missing"


# --------------------------------------------------------------------------- #
# rule parsing
# --------------------------------------------------------------------------- #
def test_parse_rule_full_document():
    rule = parse_rule(
        {
            "rule": "r1",
            "when": {"intent_any": ["a"]},
            "then": {
                "forbid_tool": "refund_order",
                "allow_tool": ["query_order"],
                "idempotent": {"tool": "refund_order", "key": "${order_id}"},
            },
            "where": {"ci": True, "runtime": False},
            "severity": "warning",
        }
    )
    assert rule.then.forbid_tool == ("refund_order",)
    assert rule.then.idempotent[0].key == "${order_id}"
    assert rule.is_runtime is False


@pytest.mark.parametrize(
    "document",
    [
        {"then": {"forbid_tool": "x"}},  # no rule name
        {"rule": "r"},  # no then
        {"rule": "r", "then": {}},  # empty then
        {"rule": "r", "then": {"forbid_tool": "x"}, "severity": "nope"},
        {"rule": "r", "then": {"idempotent": {"tool": "x"}}},  # missing key
    ],
)
def test_parse_rule_rejects_malformed_documents(document):
    from agentguard.rules import RuleError

    with pytest.raises(RuleError):
        parse_rule(document)
