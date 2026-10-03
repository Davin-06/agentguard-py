"""Tests for the two executors: CI gate and runtime guard."""

from __future__ import annotations

import json
from pathlib import Path

from agentguard.executors.ci import run_gate
from agentguard.executors.runtime import RuntimeGuard, ToolBlocked
from agentguard.idempotency import FileStore, MemoryStore
from agentguard.rules import IdempotencyRule, Rule, Then, Where

REFUND_RULE = Rule(
    name="refund-safety",
    then=Then(
        forbid_tool=("refund_order",),
        allow_tool=("query_order", "get_logistics"),
        idempotent=(IdempotencyRule("refund_order", "${order_id}"),),
    ),
    where=Where(ci=True, runtime=True),
    severity="critical",
)


# --------------------------------------------------------------------------- #
# CI executor
# --------------------------------------------------------------------------- #
def test_gate_fails_on_incident(tmp_path, refund_trace):
    trace_file = tmp_path / "trace.json"
    trace_file.write_text(
        json.dumps(
            {
                "trace_id": refund_trace.trace_id,
                "steps": [
                    {
                        "turn": s.turn,
                        "user": s.user,
                        "tool_calls": [
                            {"name": c.name, "args": c.args} for c in s.tool_calls
                        ],
                        "facts": list(s.facts),
                    }
                    for s in refund_trace.steps
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    rule_file = tmp_path / "rule.yaml"
    rule_file.write_text(
        """
rule: refund-safety
when: {intent_any: ["物流", "订单"]}
then:
  forbid_tool: refund_order
where: {ci: true, runtime: true}
severity: critical
""",
        encoding="utf-8",
    )

    outcome = run_gate(rule_file, trace_file, fail_on="critical")
    assert outcome.exit_code == 1
    assert not outcome.passed
    assert len(outcome.blocking_violations()) >= 1


def test_gate_passes_with_fail_on_none(tmp_path, refund_trace):
    trace_file = tmp_path / "trace.json"
    trace_file.write_text(
        json.dumps(
            {
                "trace_id": "x",
                "steps": [
                    {
                        "turn": 1,
                        "user": "订单到哪了",
                        "tool_calls": [{"name": "refund_order", "args": {"order_id": "A"}}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rule_file = tmp_path / "rule.yaml"
    rule_file.write_text(
        "rule: r\nthen: {forbid_tool: refund_order}\nseverity: critical\n",
        encoding="utf-8",
    )

    outcome = run_gate(rule_file, trace_file, fail_on="none")
    assert outcome.exit_code == 0


def test_ci_only_rule_is_skipped_by_runtime(refund_trace):
    """A rule with where.runtime=false must not be loaded into the guard."""
    ci_only = Rule(
        name="ci-only",
        then=Then(forbid_tool=("refund_order",)),
        where=Where(ci=True, runtime=False),
    )
    guard = RuntimeGuard([ci_only])
    assert guard.rules == []
    assert guard.check("refund_order", {}).allowed


# --------------------------------------------------------------------------- #
# Runtime guard
# --------------------------------------------------------------------------- #
def test_runtime_blocks_forbidden_tool():
    guard = RuntimeGuard([REFUND_RULE])
    decision = guard.check("refund_order", {"order_id": "A1001"}, user="我的订单到哪了")
    assert not decision.allowed
    assert decision.rule == "refund-safety"
    assert decision.severity == "critical"


def test_runtime_allows_safe_tool():
    guard = RuntimeGuard([REFUND_RULE])
    assert guard.check("query_order", {"order_id": "A"}, user="订单状态").allowed


def test_runtime_blocks_duplicate_transaction():
    guard = RuntimeGuard([REFUND_RULE])
    first = guard.check("refund_order", {"order_id": "A1"}, user="订单")
    assert not first.allowed  # forbid_tool fires first

    # Use a rule with idempotency only, so we isolate the ledger behaviour.
    idem_only = Rule(
        name="idem",
        then=Then(idempotent=(IdempotencyRule("refund_order", "${order_id}"),)),
    )
    guard2 = RuntimeGuard([idem_only])
    assert guard2.check("refund_order", {"order_id": "A1"}).allowed
    guard2.record("refund_order", {"order_id": "A1"})
    second = guard2.check("refund_order", {"order_id": "A1"})
    assert not second.allowed
    assert "duplicate transaction" in second.reason

    # A different order id is still fine.
    assert guard2.check("refund_order", {"order_id": "A2"}).allowed


def test_runtime_execute_raises_and_records_audit():
    guard = RuntimeGuard([REFUND_RULE])
    called = {"n": 0}

    def do_refund():
        called["n"] += 1
        return "refunded"

    try:
        guard.execute("refund_order", {"order_id": "A"}, do_refund, user="订单")
    except ToolBlocked as exc:
        assert exc.decision.rule == "refund-safety"
    else:  # pragma: no cover
        raise AssertionError("expected ToolBlocked")

    assert called["n"] == 0  # the tool never ran
    assert guard.audit[0]["allowed"] is False


def test_runtime_execute_runs_allowed_tool_and_records():
    guard = RuntimeGuard([REFUND_RULE])
    result = guard.execute("query_order", {"order_id": "A"}, lambda: "ok", user="订单")
    assert result == "ok"


def test_intent_gate_applies_at_runtime():
    """A rule scoped to an intent must not fire outside that intent."""
    rule = Rule(
        name="scoped",
        then=Then(forbid_tool=("refund_order",)),
        when=__import__("agentguard.rules", fromlist=["When"]).When(
            intent_any=("物流",)
        ),
    )
    guard = RuntimeGuard([rule])
    assert guard.check("refund_order", {}, user="物流到哪了").allowed is False
    assert guard.check("refund_order", {}, user="我要退款").allowed is True


def test_file_store_persists_across_instances(tmp_path):
    path = tmp_path / "ledger.json"
    store = FileStore(path)
    store.record("k")
    assert store.seen("k")

    reloaded = FileStore(path)
    assert reloaded.seen("k")
    assert reloaded.count("k") == 1


def test_memory_store_counts():
    store = MemoryStore()
    assert not store.seen("a")
    store.record("a")
    store.record("a")
    assert store.count("a") == 2


def test_guarded_tools_introspection():
    guard = RuntimeGuard([REFUND_RULE])
    mapping = guard.guarded_tools()
    assert "refund_order" in mapping
    assert "refund-safety" in mapping["refund_order"]


# --------------------------------------------------------------------------- #
# Report shapes
# --------------------------------------------------------------------------- #
def test_reports_render_without_error(tmp_path, refund_trace):
    from agentguard.engine import evaluate_all
    from agentguard.report import render_markdown, render_terminal, to_dict

    runs = [(refund_trace, evaluate_all([REFUND_RULE], refund_trace))]
    assert "FAIL" in render_terminal(runs)
    assert "❌ FAIL" in render_markdown(runs)
    payload = to_dict(runs)
    assert payload["summary"]["passed"] is False
    assert payload["runs"][0]["trace_id"] == refund_trace.trace_id
