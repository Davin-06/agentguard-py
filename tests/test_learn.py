"""Tests for trace -> candidate rule derivation (the human-in-the-loop step)."""

from __future__ import annotations

from agentguard.learn import is_destructive, learn_from_trace
from agentguard.models import Step, ToolCall, Trace


def test_is_destructive_classification():
    assert is_destructive("refund_order")
    assert is_destructive("delete_user")
    assert not is_destructive("query_order")
    assert not is_destructive("get_logistics")


def test_learn_proposes_forbidden_tool(refund_trace):
    candidates = learn_from_trace(refund_trace)
    names = [c.rule.name for c in candidates]
    assert any("forbid" in n for n in names)

    forbidden = next(c for c in candidates if "forbid" in c.rule.name)
    assert "refund_order" in forbidden.rule.then.forbid_tool
    assert forbidden.rule.source_trace == refund_trace.trace_id
    assert forbidden.needs_review is True
    assert 0.0 < forbidden.confidence <= 0.95


def test_learn_does_not_invent_a_closed_allow_list(refund_trace):
    """`allow_tool` is a *closed* list at runtime: it refuses every tool not
    named.  The learner only saw the tools this one trace used, so emitting it
    would block tools the agent legitimately has -- `refund_incident` yields
    only `query_order`, which would refuse `get_logistics`.

    It used to emit both `forbid_tool` and `allow_tool`, which also made every
    bad call appear twice in the report (once per assertion).
    """
    forbidden = next(c for c in learn_from_trace(refund_trace) if "forbid" in c.rule.name)
    assert forbidden.rule.then.allow_tool == ()
    # The description must point a human at the manual step instead of
    # pretending the generated rule is complete.
    assert "allow_tool" in forbidden.rule.description


def test_learn_proposes_idempotency(refund_trace):
    candidates = learn_from_trace(refund_trace)
    idem = [c for c in candidates if "idempotent" in c.rule.name]
    assert idem, "duplicate refund should produce an idempotency candidate"
    spec = idem[0].rule.then.idempotent[0]
    assert spec.tool == "refund_order"
    assert spec.key == "${order_id}"
    assert idem[0].confidence >= 0.8  # key ends with "id"


def test_learn_is_advisory_only(refund_trace):
    """Nothing may auto-apply: every candidate must be flagged for review."""
    assert all(c.needs_review for c in learn_from_trace(refund_trace))


def test_learn_on_clean_trace_yields_nothing_destructive(clean_trace):
    candidates = learn_from_trace(clean_trace)
    assert not [c for c in candidates if "forbid" in c.rule.name]


def test_learn_call_budget_for_loops():
    calls = tuple(ToolCall("query_order", {"order_id": "A"}, turn=i) for i in range(1, 10))
    trace = Trace(trace_id="loop", steps=(Step(turn=1, user="订单", tool_calls=calls),))
    candidates = learn_from_trace(trace, max_calls_before_flag=6)
    budget = [c for c in candidates if "call-budget" in c.rule.name]
    assert budget
    assert budget[0].rule.then.max_tool_calls == 6
    assert budget[0].rule.severity == "warning"


def test_learn_confidence_drops_when_intent_unclear():
    """A destructive call with no readable intent is a weaker signal."""
    trace = Trace(
        trace_id="vague",
        steps=(
            Step(
                turn=1,
                user="嗯",
                tool_calls=(ToolCall("refund_order", {"order_id": "A"}, turn=1),),
            ),
        ),
    )
    candidates = learn_from_trace(trace)
    forbidden = next(c for c in candidates if "forbid" in c.rule.name)
    assert forbidden.confidence < 0.7


def test_learned_rule_is_actually_enforceable(refund_trace):
    """The whole point: a learned rule must work in both executors."""
    from agentguard.engine import evaluate
    from agentguard.executors.runtime import RuntimeGuard

    candidate = next(
        c for c in learn_from_trace(refund_trace) if "forbid" in c.rule.name
    )
    result = evaluate(candidate.rule, refund_trace)
    assert not result.passed

    guard = RuntimeGuard([candidate.rule])
    decision = guard.check("refund_order", {"order_id": "A1001"}, user=refund_trace.user_text())
    assert not decision.allowed


def test_learned_intent_never_contains_destructive_vocabulary(refund_trace):
    """`when.intent_any` describes the *read-only* situation the call was wrong
    in.  A word like `退款` is the opposite -- it is how a user asks for the
    destructive action, so a rule gated on it fires when someone requests a
    refund and then forbids the refund.

    `refund_incident` used to learn `intent_any: [..., 退款]` because turn 4
    says "我没说要退款啊" -- a negation that tokenises like a request.
    """
    from agentguard.learn import HintConfig, is_destructive

    forbidden = next(c for c in learn_from_trace(refund_trace) if "forbid" in c.rule.name)
    assert "退款" not in forbidden.rule.when.intent_any, list(
        forbidden.rule.when.intent_any
    )
    # Nothing in the gate may describe a destructive action.
    vocab = HintConfig().destructive
    offenders = [
        t for t in forbidden.rule.when.intent_any if any(v in t for v in vocab)
    ]
    assert not offenders, offenders
    assert is_destructive("refund_order")  # sanity: the vocabulary is non-empty


def test_learned_rule_still_fires_on_the_read_only_intent(refund_trace):
    """Dropping destructive words must not empty the gate -- an empty `when`
    means "always applies", which is a different (and far too broad) rule."""
    forbidden = next(c for c in learn_from_trace(refund_trace) if "forbid" in c.rule.name)
    assert forbidden.rule.when.intent_any, "the learned rule lost its gate entirely"


def test_learned_rule_does_not_block_a_legitimate_refund_request():
    """The behavioural consequence of the token bug.

    A user who plainly asks for a refund must not trip a rule learned from a
    read-only conversation -- otherwise the generated ruleset blocks real
    business.
    """
    from agentguard.engine import evaluate
    from agentguard.models import Step, ToolCall, Trace

    trace = Trace(
        trace_id="legit-refund",
        steps=(
            Step(
                turn=1,
                user="我要退款",
                tool_calls=(ToolCall("refund_order", {"order_id": "A"}, turn=1),),
            ),
        ),
    )
    # Learn from the *incident*, then replay it against the legitimate request.
    incident = Trace(
        trace_id="incident",
        steps=(
            Step(
                turn=1,
                user="帮我看下物流",
                tool_calls=(ToolCall("query_order", {"order_id": "A"}, turn=1),),
            ),
            Step(
                turn=2,
                user="等等，我没说要退款啊？",
                tool_calls=(ToolCall("refund_order", {"order_id": "A"}, turn=2),),
            ),
        ),
    )
    forbidden = next(c for c in learn_from_trace(incident) if "forbid" in c.rule.name)
    assert evaluate(forbidden.rule, trace).violations == (), [
        v.message for v in evaluate(forbidden.rule, trace).violations
    ]
