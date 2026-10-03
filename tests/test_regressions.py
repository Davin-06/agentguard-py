"""Regression tests for bugs found in the 2026-10-03 audit.

Every test here corresponds to a defect that was actually observed, not a
hypothetical.  Keep them: each one was silent, i.e. the code looked correct and
produced wrong behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentguard.executors.ci import GateOutcome, _check_baseline, run_gate
from agentguard.executors.runtime import RuntimeGuard, ToolBlocked
from agentguard.idempotency import FileStore
from agentguard.exporters.promptfoo import export_promptfoo
from agentguard.models import Step, ToolCall, Trace
from agentguard.rules import (
    IdempotencyRule,
    Rule,
    RuleError,
    Then,
    When,
    Where,
    dump_rule,
    dump_rules,
    load_rule,
    load_rules,
    parse_rule,
    validate_rules,
)
from agentguard.trace import parse_trace


# --------------------------------------------------------------------------- #
# BUG 1: `learn --apply` wrote a multi-document YAML that could not be loaded.
# `yaml.safe_load` raises ComposerError on multi-doc streams, so the learned
# file was completely unreadable -- the learn -> run workflow was broken.
# --------------------------------------------------------------------------- #
def test_multi_document_rule_file_loads_all_rules(tmp_path):
    r1 = Rule(name="r1", then=Then(forbid_tool=("a",)), where=Where(True, True))
    r2 = Rule(name="r2", then=Then(forbid_tool=("b",)), where=Where(True, True))
    path = tmp_path / "learned.yaml"
    path.write_text(dump_rules([r1, r2]), encoding="utf-8")

    loaded = load_rules(path)
    assert [r.name for r in loaded] == ["r1", "r2"]


def test_load_rule_refuses_multi_document_instead_of_truncating(tmp_path):
    r1 = Rule(name="r1", then=Then(forbid_tool=("a",)))
    r2 = Rule(name="r2", then=Then(forbid_tool=("b",)))
    path = tmp_path / "two.yaml"
    path.write_text(dump_rules([r1, r2]), encoding="utf-8")

    with pytest.raises(RuleError, match="2 rule documents"):
        load_rule(path)


def test_learn_apply_output_is_reloadable(tmp_path):
    """The full learn --apply -> load_rules round trip."""
    from agentguard.cli import main

    assert main(["init", "--dir", str(tmp_path)]) == 0
    trace = tmp_path / "examples" / "refund_incident.json"
    out = tmp_path / "rules" / "learned.yaml"

    assert main(["learn", "--trace", str(trace), "--out", str(out), "--apply"]) == 0
    loaded = load_rules(out)
    assert loaded, "learned file must be loadable"
    assert all(r.then.forbid_tool or r.then.idempotent for r in loaded)


def test_learned_intent_any_is_keywords_not_whole_sentences(tmp_path):
    """A whole sentence can never match again, so such a rule is dead on arrival.

    `intent_any` is a substring test against the conversation text.  The old
    tokenizer returned the entire user sentence for Chinese input, so every
    learned rule silently never fired.
    """
    from agentguard.cli import main

    assert main(["init", "--dir", str(tmp_path)]) == 0
    trace = tmp_path / "examples" / "refund_incident.json"
    out = tmp_path / "rules" / "learned.yaml"
    assert main(["learn", "--trace", str(trace), "--out", str(out), "--apply"]) == 0

    (rule,) = [r for r in load_rules(out) if r.when.intent_any]
    assert rule.when.intent_any, "a learned rule should carry intent keywords"
    for token in rule.when.intent_any:
        assert len(token) <= 4, f"intent token looks like a whole sentence: {token!r}"
        assert "，" not in token and "。" not in token
    # And the rule must actually activate on a conversation phrased that way.
    assert "物流" in rule.when.intent_any


def test_directory_loading_spans_multiple_documents(tmp_path):
    d = tmp_path / "rules"
    d.mkdir()
    (d / "a.yaml").write_text(
        dump_rules(
            [
                Rule(name="a1", then=Then(forbid_tool=("x",))),
                Rule(name="a2", then=Then(forbid_tool=("y",))),
            ]
        ),
        encoding="utf-8",
    )
    (d / "b.yaml").write_text(
        dump_rule(Rule(name="b1", then=Then(forbid_tool=("z",)))), encoding="utf-8"
    )
    assert [r.name for r in load_rules(d)] == ["a1", "a2", "b1"]


# --------------------------------------------------------------------------- #
# BUG 2: `allow_tool` was enforced in CI but silently ignored at runtime,
# which broke the project's central claim of "one rule, two executors".
# --------------------------------------------------------------------------- #
ALLOW_RULE = Rule(
    name="allowlist",
    then=Then(allow_tool=("query_order", "get_logistics")),
    where=Where(ci=True, runtime=True),
)


def test_runtime_enforces_allow_list():
    guard = RuntimeGuard([ALLOW_RULE])
    assert guard.check("query_order", {}, user="订单").allowed
    decision = guard.check("delete_everything", {}, user="订单")
    assert not decision.allowed
    assert "outside the allow-list" in decision.reason


def test_allow_list_enforcement_can_be_disabled():
    guard = RuntimeGuard([ALLOW_RULE], enforce_allow_list=False)
    assert guard.check("delete_everything", {}, user="订单").allowed


def test_ci_and_runtime_agree_on_allow_list(refund_trace):
    """The same rule must reach the same verdict in both executors."""
    from agentguard.engine import evaluate

    ci_result = evaluate(ALLOW_RULE, refund_trace)
    assert not ci_result.passed  # refund_order is outside the list

    guard = RuntimeGuard([ALLOW_RULE])
    assert not guard.check("refund_order", {"order_id": "A1001"}, user="订单").allowed


# --------------------------------------------------------------------------- #
# BUG 3: chat-completions shaped messages ({"role","content"}) lost all text,
# so intent matching and memory assertions silently saw empty strings.
# --------------------------------------------------------------------------- #
def test_role_content_messages_are_parsed():
    trace = parse_trace(
        {
            "trace_id": "chat",
            "messages": [
                {"role": "user", "content": "我上周那单还没到"},
                {"role": "assistant", "content": "我查一下"},
            ],
        }
    )
    assert trace.steps[0].user == "我上周那单还没到"
    assert trace.steps[1].assistant == "我查一下"


def test_role_content_messages_carry_tool_calls():
    trace = parse_trace(
        {
            "trace_id": "chat",
            "messages": [
                {"role": "user", "content": "查订单"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "refund_order", "arguments": '{"order_id":"A1"}'}}
                    ],
                },
            ],
        }
    )
    call = trace.tool_calls()[0]
    assert call.name == "refund_order"
    assert call.args == {"order_id": "A1"}


def test_role_content_intent_matching_works():
    """Regression: intent gate saw empty text, so rules never applied."""
    from agentguard.engine import rule_applies

    trace = parse_trace(
        {
            "trace_id": "chat",
            "messages": [
                {"role": "user", "content": "帮我看下物流"},
                {"role": "assistant", "content": "好的"},
            ],
        }
    )
    rule = Rule(
        name="r",
        then=Then(forbid_tool=("refund_order",)),
        when=When(intent_any=("物流",)),
    )
    assert rule_applies(rule, trace)


def test_multimodal_content_parts_are_flattened():
    trace = parse_trace(
        {
            "trace_id": "mm",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "看看这个订单"}],
                }
            ],
        }
    )
    assert "看看这个订单" in trace.steps[0].user


def test_tool_role_messages_do_not_pollute_user_text():
    trace = parse_trace(
        {
            "trace_id": "chat",
            "messages": [
                {"role": "user", "content": "查物流"},
                {"role": "tool", "content": '{"raw": "payload"}'},
            ],
        }
    )
    assert "payload" not in trace.user_text()


# --------------------------------------------------------------------------- #
# BUG 4: the promptfoo exporter emitted JS referencing `context.vars.__trajectory`,
# a variable promptfoo never provides (the real field is `context.trace`).
# Those assertions would have failed at runtime while looking correct.
# --------------------------------------------------------------------------- #
def test_promptfoo_export_contains_no_fake_context_vars():
    result = export_promptfoo([ALLOW_RULE])
    text = json.dumps(result.config, ensure_ascii=False)
    assert "__trajectory" not in text
    assert "context.vars" not in text


def test_promptfoo_export_reports_unsupported_constraints():
    result = export_promptfoo([ALLOW_RULE])
    assert any("allow_tool" in item for item in result.unsupported)
    assert "agentguard_unsupported" in result.config["metadata"]


def test_promptfoo_export_skips_rules_it_cannot_express():
    """A rule with nothing expressible must not become a silently-passing test."""
    only_idem = Rule(
        name="idem-only",
        then=Then(idempotent=(IdempotencyRule("refund_order", "${order_id}"),)),
    )
    result = export_promptfoo([only_idem])
    assert result.config["tests"] == []
    assert result.unsupported


def test_promptfoo_export_still_emits_native_assertions():
    rule = Rule(
        name="native",
        then=Then(forbid_tool=("refund_order",), require_tool=("query_order",), max_tool_calls=5),
    )
    result = export_promptfoo([rule])
    types = {a["type"] for a in result.config["tests"][0]["assert"]}
    assert types == {
        "not-trajectory:tool-used",
        "trajectory:tool-used",
        "trajectory:step-count",
    }


# --------------------------------------------------------------------------- #
# BUG 5: baseline comparison mixed "total violations" (baseline file) with
# "blocking violations" (current run), producing false regressions whenever
# --fail-on differed between runs.
# --------------------------------------------------------------------------- #
def _outcome(threshold: str) -> GateOutcome:
    from agentguard.engine import evaluate_all
    from agentguard.models import Step, ToolCall, Trace

    rule = Rule(
        name="r",
        then=Then(forbid_tool=("refund_order",)),
        severity="critical",
    )
    warn_rule = Rule(
        name="w", then=Then(forbid_tool=("noise",)), severity="warning"
    )
    trace = Trace(
        trace_id="t",
        steps=(
            Step(
                turn=1,
                user="x",
                tool_calls=(
                    ToolCall("refund_order", {}, turn=1),
                    ToolCall("noise", {}, turn=1),
                ),
            ),
        ),
    )
    return GateOutcome(
        runs=[(trace, evaluate_all([rule, warn_rule], trace))],
        exit_code=0,
        threshold=threshold,
    )


def test_baseline_uses_total_not_blocking(tmp_path):
    outcome = _outcome(threshold="critical")
    assert len(outcome.blocking_violations()) == 1
    assert len(outcome.total_violations()) == 2

    baseline = tmp_path / "base.json"
    baseline.write_text(json.dumps({"summary": {"violations": 2}}), encoding="utf-8")
    # 2 total vs 2 total -> no regression (the old code compared 1 vs 2 -> false alarm)
    assert _check_baseline(baseline, outcome) is None


def test_baseline_detects_real_regression(tmp_path):
    outcome = _outcome(threshold="critical")
    baseline = tmp_path / "base.json"
    baseline.write_text(json.dumps({"summary": {"violations": 1}}), encoding="utf-8")
    assert _check_baseline(baseline, outcome) is not None


def test_baseline_missing_file_is_not_an_error(tmp_path):
    """No baseline file means the project has not adopted one yet."""
    outcome = _outcome(threshold="critical")
    assert _check_baseline(tmp_path / "nope.json", outcome) is None


def test_baseline_corrupt_file_is_an_error(tmp_path):
    """This test used to assert the opposite (`... is None`), which pinned the
    bug: a corrupt baseline was treated as "no baseline", so the regression
    check silently did nothing.  See BUG 47."""
    from agentguard.executors.ci import BaselineError

    outcome = _outcome(threshold="critical")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(BaselineError):
        _check_baseline(bad, outcome)


def test_baseline_accepts_the_flat_violations_form(tmp_path):
    """`{"violations": N}` is the obvious hand-written form and used to be
    ignored in silence -- the number was present but never compared, so a
    regression on a clean baseline was never reported."""
    outcome = _outcome(threshold="critical")
    baseline = tmp_path / "flat.json"
    baseline.write_text(json.dumps({"violations": 1}), encoding="utf-8")
    note = _check_baseline(baseline, outcome)
    assert note is not None, "flat-format baseline was ignored, not compared"


def test_baseline_rejects_a_file_without_a_count(tmp_path):
    from agentguard.executors.ci import BaselineError

    outcome = _outcome(threshold="critical")
    baseline = tmp_path / "nokeys.json"
    baseline.write_text(json.dumps({"something": "else"}), encoding="utf-8")
    with pytest.raises(BaselineError):
        _check_baseline(baseline, outcome)


def test_baseline_rejects_an_empty_file(tmp_path):
    from agentguard.executors.ci import BaselineError

    outcome = _outcome(threshold="critical")
    baseline = tmp_path / "empty.json"
    baseline.write_text("", encoding="utf-8")
    with pytest.raises(BaselineError):
        _check_baseline(baseline, outcome)


def test_baseline_accepts_the_json_command_output_verbatim(tmp_path):
    """`agentguard run --format json > baseline.json` must round-trip without
    the user hand-editing the shape."""
    import os

    from agentguard.cli import main

    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "baseline.json"

    cwd = os.getcwd()
    try:
        os.chdir(repo)
        main(
            [
                "run",
                "--traces",
                "examples/refund_incident/trace.json",
                "--format",
                "json",
                "--out",
                str(tmp_path),
            ]
        )
    finally:
        os.chdir(cwd)

    produced = tmp_path / "agentguard-results.json"
    assert produced.exists()
    out.write_text(produced.read_text(encoding="utf-8"), encoding="utf-8")

    outcome = _outcome(threshold="critical")
    # Same violation count as recorded -> no regression reported, and crucially
    # no error, which is what proves the shape was understood.
    assert _check_baseline(out, outcome) is None


def test_gate_with_baseline_sets_exit_code(tmp_path, refund_trace):
    trace_file = tmp_path / "t.json"
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
    rule_file = tmp_path / "r.yaml"
    rule_file.write_text(
        "rule: r\nthen: {forbid_tool: refund_order}\nseverity: critical\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "b.json"
    baseline.write_text(json.dumps({"summary": {"violations": 0}}), encoding="utf-8")

    outcome = run_gate(rule_file, trace_file, fail_on="none", baseline=baseline)
    assert outcome.exit_code == 1  # regression vs baseline despite fail_on=none
    assert outcome.notes


# --------------------------------------------------------------------------- #
# BUG 6/7: rules that cannot work at runtime were loaded without any signal.
# --------------------------------------------------------------------------- #
def test_guard_warns_on_non_enforced_severity():
    rule = Rule(
        name="info-rule",
        then=Then(forbid_tool=("x",)),
        severity="info",
        where=Where(True, True),
    )
    guard = RuntimeGuard([rule])
    assert any("severity" in w for w in guard.warnings)
    assert guard.check("x", {}).allowed  # documented behaviour: not enforced


def test_guard_warns_on_turn_range_rule():
    rule = Rule(
        name="turn-rule",
        then=Then(forbid_tool=("x",)),
        when=When(turn_range=(5, 10)),
        where=Where(True, True),
    )
    guard = RuntimeGuard([rule])
    assert any("turn_range" in w for w in guard.warnings)


def test_guard_warns_on_memory_rule_marked_runtime():
    from agentguard.rules import MemoryRule

    rule = Rule(
        name="mem-rule",
        then=Then(memory_persistence=(MemoryRule(fact="F", must_recall_at=(5,)),)),
        where=Where(ci=True, runtime=True),
    )
    guard = RuntimeGuard([rule])
    assert any("memory_persistence" in w for w in guard.warnings)


def test_guard_has_no_warnings_for_healthy_rules():
    guard = RuntimeGuard(
        [Rule(name="ok", then=Then(forbid_tool=("x",)), where=Where(True, True))]
    )
    assert guard.warnings == []


# --------------------------------------------------------------------------- #
# Static rule validation -- surfacing silent misconfigurations.
# --------------------------------------------------------------------------- #
def test_validate_warns_on_require_tool_marked_runtime():
    """require_tool describes whole-trace completeness; per-call it is useless."""
    from agentguard.rules import validate_rules

    rule = Rule(
        name="req",
        then=Then(require_tool=("query_order",)),
        where=Where(ci=True, runtime=True),
    )
    assert any("require_tool" in w for w in validate_rules([rule]))


def test_validate_warns_when_rule_never_runs():
    from agentguard.rules import validate_rules

    rule = Rule(
        name="dead",
        then=Then(forbid_tool=("x",)),
        where=Where(ci=False, runtime=False),
    )
    assert any("永远不会执行" in w for w in validate_rules([rule]))


def test_validate_warns_on_duplicate_rule_names():
    from agentguard.rules import validate_rules

    rules = [
        Rule(name="dup", then=Then(forbid_tool=("x",))),
        Rule(name="dup", then=Then(forbid_tool=("y",))),
    ]
    assert any("出现 2 次" in w for w in validate_rules(rules))


def test_validate_clean_rules_produce_no_warnings():
    from agentguard.rules import validate_rules

    rules = [
        Rule(name="a", then=Then(forbid_tool=("x",)), where=Where(True, True)),
        Rule(
            name="b",
            then=Then(forbid_tool=("y",)),
            where=Where(ci=True, runtime=False),
        ),
    ]
    assert validate_rules(rules) == []


def test_ci_run_prints_rule_warnings(tmp_path, capsys):
    """`agentguard run` must surface misconfigurations, not stay silent."""
    from agentguard.cli import main

    assert main(["init", "--dir", str(tmp_path)]) == 0
    (tmp_path / "rules" / "bad.yaml").write_text(
        "rule: bad\nthen: {require_tool: [x]}\nwhere: {ci: true, runtime: true}\n",
        encoding="utf-8",
    )
    capsys.readouterr()
    main(["run", "--rules", str(tmp_path / "rules"), "--traces", str(tmp_path / "traces")])
    captured = capsys.readouterr()
    assert "require_tool" in captured.err


def test_report_label_is_rule_trace_pairs(refund_trace):
    """The denominator is rules x traces, so the label must say so."""
    from agentguard.engine import evaluate_all
    from agentguard.report import render_terminal

    rule = Rule(name="r", then=Then(forbid_tool=("refund_order",)))
    runs = [(refund_trace, evaluate_all([rule], refund_trace))]
    assert "规则×用例命中" in render_terminal(runs)


# =========================================================================== #
# Second audit pass (2026-10-03, later) -- seven more defects.
# =========================================================================== #


# --------------------------------------------------------------------------- #
# BUG 8: unknown keys in `when` / `then` / `where` were silently ignored.
# `when.intent` (used in the original design doc!) parsed to an EMPTY When,
# which means "applies unconditionally" -- the exact opposite of the intent.
# --------------------------------------------------------------------------- #
def test_unknown_key_in_when_is_rejected_with_a_suggestion():
    with pytest.raises(RuleError, match="intent"):
        parse_rule(
            {
                "rule": "typo",
                "when": {"intent": "order_status_query"},
                "then": {"forbid_tool": "refund_order"},
            }
        )


def test_unknown_key_in_then_is_rejected():
    with pytest.raises(RuleError, match="forbit_tool"):
        parse_rule(
            {"rule": "typo", "then": {"forbid_tool": "x", "forbit_tool": "y"}}
        )


def test_unknown_key_in_where_is_rejected():
    with pytest.raises(RuleError):
        parse_rule(
            {
                "rule": "typo",
                "then": {"forbid_tool": "x"},
                "where": {"ci": True, "runtim": True},
            }
        )


def test_unknown_key_in_nested_idempotent_is_rejected():
    with pytest.raises(RuleError):
        parse_rule(
            {
                "rule": "typo",
                "then": {"idempotent": {"tool": "t", "keys": "${id}"}},
            }
        )


def test_valid_keys_still_parse():
    rule = parse_rule(
        {
            "version": 1,
            "rule": "ok",
            "description": "d",
            "source_trace": "t1",
            "when": {"intent_any": ["x"], "turn_range": [1, 5]},
            "then": {
                "forbid_tool": ["a"],
                "allow_tool": ["b"],
                "require_tool": ["c"],
                "max_tool_calls": 3,
                "idempotent": {"tool": "d", "key": "${k}"},
                "memory_persistence": {"fact": "f", "inject_at": 1, "must_recall_at": [2]},
            },
            "where": {"ci": True, "runtime": False},
            "severity": "warning",
        }
    )
    assert rule.name == "ok"


def test_reversed_turn_range_is_rejected():
    with pytest.raises(RuleError, match="永远不会生效"):
        parse_rule(
            {
                "rule": "bad-range",
                "when": {"turn_range": [10, 5]},
                "then": {"forbid_tool": "x"},
            }
        )


# --------------------------------------------------------------------------- #
# BUG 9: NeMo Colang generation was fabricated.
# `rails.tool_output` accepts only the built-in name "tool call validation";
# an unknown flow name makes NeMo SILENTLY fall back and disable every tool
# rail. The old exporter emitted `define flow agentguard ...` -- which would
# have switched off the very guardrails it claimed to configure.
# --------------------------------------------------------------------------- #
def test_nemo_flow_name_is_the_builtin_one():
    from agentguard.exporters.nemo import TOOL_OUTPUT_FLOW, export_nemo

    rule = Rule(
        name="r", then=Then(allow_tool=("query_order",)), where=Where(True, True)
    )
    config = export_nemo([rule]).config
    assert config["rails"]["tool_output"]["flows"] == [TOOL_OUTPUT_FLOW]
    assert TOOL_OUTPUT_FLOW == "tool call validation"


def test_nemo_export_has_no_invented_colang():
    from agentguard.exporters.nemo import render_nemo

    rule = Rule(name="r", then=Then(forbid_tool=("refund_order",)))
    text = render_nemo([rule])
    assert "define flow" not in text


def test_nemo_export_states_its_prerequisites():
    from agentguard.exporters.nemo import export_nemo

    rule = Rule(name="r", then=Then(allow_tool=("q",)))
    result = export_nemo([rule])
    joined = " ".join(result.requirements)
    assert "IORAILS" in joined
    assert "Colang 1.0" in joined


# --------------------------------------------------------------------------- #
# BUG 10: `_learn_call_budget` used a flat confidence of 0.5, below the CLI's
# default --min-confidence of 0.6 -- so the candidate was never written and
# the feature was silently dead.
# --------------------------------------------------------------------------- #
def test_call_budget_candidate_clears_the_default_threshold():
    from agentguard.learn import learn_from_trace

    calls = tuple(
        ToolCall("query_order", {"order_id": "A"}, turn=i) for i in range(1, 12)
    )
    trace = Trace(trace_id="loop", steps=(Step(turn=1, user="订单", tool_calls=calls),))
    budget = [
        c for c in learn_from_trace(trace) if "call-budget" in c.rule.name
    ]
    assert budget, "call-budget candidate must be produced"
    assert budget[0].confidence >= 0.6


def test_call_budget_confidence_scales_with_overshoot():
    from agentguard.learn import learn_from_trace

    def conf(n: int) -> float:
        calls = tuple(ToolCall("q", {}, turn=i) for i in range(1, n + 1))
        trace = Trace(trace_id="l", steps=(Step(turn=1, user="x", tool_calls=calls),))
        return next(
            c.confidence for c in learn_from_trace(trace) if "call-budget" in c.rule.name
        )

    assert conf(20) > conf(7)


# --------------------------------------------------------------------------- #
# BUG 11: Chinese tool names were never recognised as destructive, so the
# whole `learn` feature was useless for Chinese-named tools.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "tool", ["退款", "退款工具", "取消订单", "删除用户", "下单", "扣款"]
)
def test_chinese_tools_are_destructive(tool):
    from agentguard.learn import is_destructive

    assert is_destructive(tool)


@pytest.mark.parametrize("tool", ["查询订单", "获取物流", "查看详情", "query_order"])
def test_chinese_readonly_tools_are_not_destructive(tool):
    from agentguard.learn import is_destructive

    assert not is_destructive(tool)


def test_hints_can_be_extended_from_a_file(tmp_path):
    from agentguard.learn import HintConfig

    path = tmp_path / "hints.yaml"
    path.write_text(
        "destructive: [核销]\nreadonly: [预览]\n", encoding="utf-8"
    )
    hints = HintConfig.from_file(path)
    assert hints.is_destructive("核销优惠券")
    assert not hints.is_destructive("预览订单")
    # built-ins still present
    assert hints.is_destructive("refund_order")
    assert hints.is_destructive("退款")


def test_hints_from_file_can_replace_builtins(tmp_path):
    from agentguard.learn import HintConfig

    path = tmp_path / "hints.yaml"
    path.write_text("destructive: [核销]\nreadonly: []\n", encoding="utf-8")
    hints = HintConfig.from_file(path, replace=True)
    assert hints.is_destructive("核销")
    assert not hints.is_destructive("refund_order")


def test_learn_with_custom_hints_detects_custom_tool():
    from agentguard.learn import HintConfig, learn_from_trace

    trace = Trace(
        trace_id="custom",
        steps=(
            Step(turn=1, user="我想看看优惠券"),
            Step(
                turn=2,
                user="嗯",
                tool_calls=(ToolCall("核销优惠券", {"id": "C1"}, turn=2),),
            ),
        ),
    )
    hints = HintConfig(destructive=("核销",), readonly=("查询", "查看"))
    found = [c for c in learn_from_trace(trace, hints=hints) if "forbid" in c.rule.name]
    assert found, "custom hint should make 核销优惠券 detectable"


# --------------------------------------------------------------------------- #
# BUG 12: FileStore wrote non-atomically. A crash mid-write left invalid JSON,
# and the loader treats unreadable content as an EMPTY ledger -- silently
# disabling every idempotency check.
# --------------------------------------------------------------------------- #
def test_file_store_write_is_atomic(tmp_path):
    from agentguard.idempotency import FileStore

    path = tmp_path / "ledger.json"
    store = FileStore(path)
    store.record("a")
    store.record("b")
    # No stray temp files left behind.
    leftovers = [p for p in tmp_path.iterdir() if p.name != "ledger.json"]
    assert leftovers == []
    assert FileStore(path).as_dict() == {"a": 1, "b": 1}


def test_file_store_refuses_to_silently_reset_on_corruption(tmp_path):
    """A corrupt ledger must not be mistaken for an empty one.

    Treating unreadable content as "no transactions recorded yet" disables every
    idempotency check without a word -- the guard would then allow a second
    charge on the same order, which is the exact incident this tool exists to
    stop.  (An earlier version of this test asserted the lenient behaviour; the
    leniency *was* the bug.)
    """
    from agentguard.idempotency import FileStore, LedgerError

    path = tmp_path / "ledger.json"
    path.write_text("{half-written", encoding="utf-8")
    with pytest.raises(LedgerError, match="损坏"):
        FileStore(path)

    # A zero-byte file is a fresh start, not corruption.
    path.write_text("", encoding="utf-8")
    assert FileStore(path).as_dict() == {}


def test_file_store_treats_a_missing_file_as_fresh(tmp_path):
    from agentguard.idempotency import FileStore

    assert FileStore(tmp_path / "never-written.json").as_dict() == {}


def test_file_store_rejects_non_object_json(tmp_path):
    from agentguard.idempotency import FileStore, LedgerError

    path = tmp_path / "ledger.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(LedgerError, match="JSON 对象"):
        FileStore(path)


def test_file_store_handles_unwritable_target_cleanly(tmp_path, monkeypatch):
    """A failed write must not leave a temp file behind."""
    from agentguard.idempotency import FileStore

    path = tmp_path / "ledger.json"
    store = FileStore(path)

    import os as _os

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(_os, "replace", boom)
    with pytest.raises(OSError):
        store.record("a")
    assert [p for p in tmp_path.iterdir()] == []


# --------------------------------------------------------------------------- #
# BUG 13: `must_recall_at` earlier than `inject_at` can never pass, and
# nothing said so.
# --------------------------------------------------------------------------- #
def test_validate_warns_when_recall_precedes_injection():
    from agentguard.rules import MemoryRule, validate_rules

    rule = Rule(
        name="m",
        then=Then(memory_persistence=(MemoryRule(fact="F", inject_at=10, must_recall_at=(3,)),)),
    )
    assert any("必然失败" in w for w in validate_rules([rule]))


def test_validate_warns_on_empty_fact():
    from agentguard.rules import MemoryRule, validate_rules

    rule = Rule(
        name="m",
        then=Then(memory_persistence=(MemoryRule(fact="   ", must_recall_at=(5,)),)),
    )
    assert any("空字符串" in w for w in validate_rules([rule]))


# --------------------------------------------------------------------------- #
# BUG 14: CLI `--arg amount=299` kept the value as a string, so the idempotency
# key differed from the same transaction arriving as JSON from CI.
# --------------------------------------------------------------------------- #
def test_cli_arg_values_are_json_coerced():
    from agentguard.cli import _coerce_arg

    assert _coerce_arg("299") == 299
    assert _coerce_arg("true") is True
    assert _coerce_arg("null") is None
    assert _coerce_arg("A1001") == "A1001"
    assert _coerce_arg('{"a": 1}') == {"a": 1}
    assert _coerce_arg("") == ""


def test_idempotency_key_is_stable_across_entry_points(tmp_path):
    """int 299 from JSON and '299' from CLI must not be two different keys."""
    from agentguard.engine import render_template
    from agentguard.idempotency import MemoryStore

    rule = Rule(
        name="idem",
        then=Then(idempotent=(IdempotencyRule("refund_order", "${order_id}"),)),
    )
    guard = RuntimeGuard([rule], store=MemoryStore())
    # Record with an int key...
    guard.record("refund_order", {"order_id": 299})
    # ...then check with the same value parsed from a CLI string.
    from agentguard.cli import _coerce_arg

    decision = guard.check("refund_order", {"order_id": _coerce_arg("299")})
    assert not decision.allowed
    assert render_template("${order_id}", {"order_id": _coerce_arg("299")}) == 299


# --------------------------------------------------------------------------- #
# BUG 15: `_slug` stripped every non-ASCII character, so a Chinese trace id
# became the literal "rule" -- all learned rules were named
# `rule-forbid-rule`, indistinguishable from each other.
# --------------------------------------------------------------------------- #
def test_slug_preserves_cjk():
    from agentguard.learn import _slug

    assert _slug("退款事故") == "退款事故"
    assert _slug("退款事故-中文工具名") == "退款事故-中文工具名"
    assert _slug("Order Flow!") == "order-flow"
    assert _slug("!!!") == "rule"  # still falls back for pure punctuation


def test_learned_rule_name_from_chinese_trace_is_meaningful():
    from agentguard.learn import learn_from_trace

    trace = Trace(
        trace_id="退款事故",
        steps=(
            Step(turn=1, user="查下物流"),
            Step(
                turn=2,
                user="嗯",
                tool_calls=(ToolCall("退款", {"订单号": "CN-1"}, turn=2),),
            ),
        ),
    )
    names = [c.rule.name for c in learn_from_trace(trace)]
    assert any("退款事故" in n for n in names)
    assert not any(n.startswith("rule-forbid-rule") for n in names)


# --------------------------------------------------------------------------- #
# BUG 16: `_guess_key` only knew English identifier names, so idempotency
# candidates were never produced for Chinese argument names like 订单号.
# --------------------------------------------------------------------------- #
def test_guess_key_recognises_chinese_identifiers():
    from agentguard.learn import _guess_key

    calls = [
        ToolCall("退款", {"订单号": "CN-1", "金额": 199}, turn=1),
        ToolCall("退款", {"订单号": "CN-1", "金额": 199}, turn=2),
    ]
    assert _guess_key(calls) == "订单号"


def test_learn_produces_idempotency_for_chinese_tools():
    from agentguard.learn import learn_from_trace

    trace = Trace(
        trace_id="退款事故",
        steps=(
            Step(turn=1, user="查下物流"),
            Step(
                turn=2,
                user="嗯",
                tool_calls=(
                    ToolCall("退款", {"订单号": "CN-1", "金额": 199}, turn=2),
                    ToolCall("退款", {"订单号": "CN-1", "金额": 199}, turn=3),
                ),
            ),
        ),
    )
    idem = [c for c in learn_from_trace(trace) if "idempotent" in c.rule.name]
    assert idem, "Chinese argument names must yield an idempotency candidate"
    assert idem[0].rule.then.idempotent[0].key == "${订单号}"


# --------------------------------------------------------------------------- #
# BUG 17: the shipped GitHub Actions workflow ran
#   `agentguard gate --rules rules --traces traces`
# but the repo contained no `traces/` directory at all.  A fresh clone would
# fail with "trace path not found: traces" -- the project's own CI was red on
# day one.  Guard the contract with a test so it cannot silently regress.
# --------------------------------------------------------------------------- #
def test_shipped_ci_baseline_exists_and_is_clean():
    repo = Path(__file__).resolve().parents[1]
    traces = repo / "traces"
    rules = repo / "rules"

    assert traces.is_dir(), "the CI workflow points at traces/; it must exist"
    assert list(traces.glob("*.json")), "traces/ must contain at least one baseline trace"

    outcome = run_gate(rules, traces, fail_on="critical")
    assert outcome.total_violations() == [], (
        "the bundled traces/ baseline is supposed to be clean -- if a rule now "
        "fires on it, either the rule or the baseline is wrong, and CI will be red"
    )
    assert outcome.exit_code == 0


def test_ci_workflow_paths_match_the_repo_layout():
    """The workflow's `--traces` argument must point at a directory that exists."""
    repo = Path(__file__).resolve().parents[1]
    workflow = repo / ".github" / "workflows" / "agentguard.yml"

    text = workflow.read_text(encoding="utf-8")
    assert "--traces traces" in text
    assert (repo / "traces").is_dir()
    assert "--rules rules" in text
    assert (repo / "rules").is_dir()


# --------------------------------------------------------------------------- #
# BUG 18 (usability audit): `_tokens` returned whole Chinese sentences.
# Because `intent_any` is a substring match, a learned rule carrying a whole
# sentence as its activation condition could never fire -- learn produced rules
# that looked right and did nothing.
# --------------------------------------------------------------------------- #
def test_intent_tokens_are_keywords_not_sentences():
    from agentguard.learn import _tokens

    assert _tokens("我上周那单还没到，帮我看下物流") == ["物流"]
    assert _tokens("对，就是那个蓝色的") == ["蓝色"]
    assert _tokens("我要退款") == ["退款"]
    tokens = _tokens("我的订单到货了吗")
    assert "订单" in tokens and "到货" in tokens


def test_intent_tokens_keep_english_words():
    from agentguard.learn import _tokens

    assert "track" in _tokens("I want to track my order")
    assert "the" not in _tokens("I want to track my order")


def test_intent_tokens_preserve_domain_characters():
    """Stopword stripping must not shred real domain words like 到货 / 退款."""
    from agentguard.learn import _tokens

    for word in ("到货", "退款", "订单", "扣款"):
        assert word in _tokens(f"我要{word}"), word


# --------------------------------------------------------------------------- #
# BUG 19 (usability audit): `export` dropped the `when` gate without saying so.
# promptfoo has no conditional-assertion primitive, so the emitted assertion
# applies unconditionally -- a rule meant for logistics conversations becomes a
# blanket ban, and legitimate refund tests start failing.  Silent widening.
# --------------------------------------------------------------------------- #
def test_promptfoo_export_flags_the_lost_when_gate():
    from agentguard.exporters.promptfoo import export_promptfoo

    rule = Rule(
        name="scoped",
        when=When(intent_any=("物流",)),
        then=Then(forbid_tool=("refund_order",)),
        where=Where(True, True),
    )
    result = export_promptfoo([rule])
    assert result.config["tests"], "the assertion itself is still expressible"
    assert any("when=" in item and "所有" in item for item in result.unsupported)


def test_nemo_export_flags_the_lost_when_gate():
    from agentguard.exporters.nemo import export_nemo

    rule = Rule(
        name="scoped",
        when=When(intent_any=("物流",)),
        then=Then(allow_tool=("query_order",)),
        where=Where(True, True),
    )
    result = export_nemo([rule])
    assert any("when=" in item for item in result.unsupported)


# --------------------------------------------------------------------------- #
# BUG 20 (usability audit): when no intent keyword can be extracted, the rule
# gets an empty `when` -- which means "always applies".  The candidate must not
# clear the default --apply threshold, or learn would silently write a rule that
# forbids the tool in *every* conversation, including legitimate ones.
# --------------------------------------------------------------------------- #
def test_unscoped_candidate_falls_below_the_apply_threshold():
    from agentguard.learn import learn_from_trace

    # A purely English-noise sentence yields no usable keyword.
    trace = Trace(
        trace_id="no-intent",
        steps=(
            Step(turn=1, user="嗯嗯嗯"),
            Step(turn=2, user="好的好的", tool_calls=(ToolCall("refund_order", {}, turn=2),)),
        ),
    )
    candidates = [c for c in learn_from_trace(trace) if c.rule.then.forbid_tool]
    assert candidates
    assert candidates[0].confidence < 0.6
    assert not candidates[0].rule.when.intent_any


# --------------------------------------------------------------------------- #
# BUG 21 (usability audit): `init` left the user with a red gate.
# It told them to run `agentguard run`, and that immediately failed, because the
# only trace it wrote was an incident.  A fresh scaffold must be green, with the
# incident kept separate.
# --------------------------------------------------------------------------- #
def test_init_scaffold_is_green_and_incident_is_separate(tmp_path):
    from agentguard.cli import main

    assert main(["init", "--dir", str(tmp_path)]) == 0

    baseline = tmp_path / "traces"
    incident = tmp_path / "examples"
    assert (baseline / "baseline-clean.json").exists()
    assert (incident / "refund_incident.json").exists()

    assert main(["run", "--rules", str(tmp_path / "rules"), "--traces", str(baseline)]) == 0
    assert main(["run", "--rules", str(tmp_path / "rules"), "--traces", str(incident)]) == 1


# --------------------------------------------------------------------------- #
# BUG 22 (usability audit): the CLI printed English assertion messages inside
# otherwise-Chinese output.  The canonical message stays English (it reaches
# production logs and ToolBlocked tracebacks), so the fix was to add a
# machine-readable `code` plus a localised rendering layer.  These tests keep
# the two in sync.
# --------------------------------------------------------------------------- #
def test_runtime_decision_carries_a_machine_readable_code():
    from agentguard.idempotency import MemoryStore

    guard = RuntimeGuard(
        [
            Rule(name="r", then=Then(forbid_tool=("bad",)), where=Where(True, True)),
            Rule(name="a", then=Then(allow_tool=("ok",)), where=Where(True, True)),
        ]
    )
    assert guard.check("bad", {}).code == "forbidden_tool"
    assert guard.check("other", {}).code == "tool_not_allowed"
    assert guard.check("ok", {}).code == "allowed"
    assert guard.check("ok", {}).reason == "no rule objected"

    idem = RuntimeGuard(
        [Rule(name="i", then=Then(idempotent=(IdempotencyRule("pay", "${oid}"),)))],
        store=MemoryStore(),
    )
    idem.record("pay", {"oid": "A1"})
    decision = idem.check("pay", {"oid": "A1"})
    assert decision.code == "duplicate_transaction"
    assert decision.detail["key"] == "A1"
    assert decision.detail["key_name"] == "oid"


def test_every_violation_kind_has_a_chinese_rendering():
    """Adding a new `kind` without a rendering must fail here.

    The kinds are scanned out of engine.py rather than hardcoded, so this guard
    keeps working when someone adds a check.
    """
    import re
    from pathlib import Path

    from agentguard.messages import violation_zh

    engine = Path(__file__).resolve().parents[1] / "agentguard" / "engine.py"
    kinds = sorted(set(re.findall(r'kind="([a-z_]+)"', engine.read_text(encoding="utf-8"))))
    assert kinds, "no violation kinds found -- did engine.py move?"

    for kind in kinds:
        fallback = f"ENGLISH FALLBACK {kind}"
        rendered = violation_zh(kind=kind, message=fallback, detail={}, turn=1, tool="t")
        assert rendered != fallback, f"kind {kind!r} has no Chinese rendering"


def test_every_decision_code_has_a_chinese_rendering():
    from agentguard.messages import decision_zh

    for code in ("allowed", "forbidden_tool", "tool_not_allowed", "duplicate_transaction"):
        fallback = f"ENGLISH FALLBACK {code}"
        rendered = decision_zh(code=code, reason=fallback, detail={}, rule="r")
        assert rendered != fallback, f"code {code!r} has no Chinese rendering"


def test_unknown_kind_falls_back_to_the_canonical_message():
    """A report that silently drops a violation is worse than an English line."""
    from agentguard.messages import decision_zh, violation_zh

    assert violation_zh(kind="brand_new", message="canonical", detail={}) == "canonical"
    assert decision_zh(code="brand_new", reason="canonical", detail={}) == "canonical"


def test_json_payload_keeps_the_canonical_message():
    """`to_dict()` is the machine-readable payload -- do not localise it.

    The template is English but may interpolate user data (a Chinese fact name,
    say), so we assert identity with the model rather than pure ASCII.
    """
    from pathlib import Path

    from agentguard.executors.ci import run_gate
    from agentguard.messages import violation_zh
    from agentguard.report import to_dict

    repo = Path(__file__).resolve().parents[1]
    outcome = run_gate(repo / "rules", repo / "examples", fail_on="critical")
    payload = to_dict(outcome.runs)

    seen = 0
    for run, (_trace, results) in zip(payload["runs"], outcome.runs):
        for rule_payload, result in zip(run["rules"], results):
            for entry, violation in zip(rule_payload["violations"], result.violations):
                assert entry["message"] == violation.message
                assert entry["kind"] == violation.kind
                # The localised form is an extra rendering, never a replacement.
                assert entry["message"] != violation_zh(
                    kind=violation.kind,
                    message=violation.message,
                    detail=violation.detail,
                    turn=violation.turn,
                    tool=violation.tool,
                )
                seen += 1
    assert seen, "expected violations in the bundled incident traces"


# --------------------------------------------------------------------------- #
# The central invariant: one rule, two executors, same verdict.
#
# The project's whole claim is that CI and the runtime guard read the *same*
# rule and therefore cannot drift apart.  Nothing was asserting it.  If CI
# flags a call and the runtime guard lets it through, the drift the tool exists
# to eliminate is back -- and it would be invisible, because both sides look
# individually correct.
# --------------------------------------------------------------------------- #
def test_ci_and_runtime_agree_on_every_blocked_tool_call():
    from pathlib import Path

    from agentguard.engine import evaluate
    from agentguard.rules import load_rules
    from agentguard.trace import load_traces

    repo = Path(__file__).resolve().parents[1]
    runtime_rules = [r for r in load_rules(repo / "rules") if r.is_runtime]
    assert runtime_rules, "expected runtime-enforced rules in rules/"
    guard = RuntimeGuard(runtime_rules)

    checked = 0
    disagreements = []
    for trace in load_traces(repo / "examples"):
        for rule in runtime_rules:
            result = evaluate(rule, trace)
            for violation in result.violations:
                if violation.kind not in ("forbidden_tool", "tool_not_allowed"):
                    continue
                if violation.tool is None:
                    continue
                decision = guard.check(
                    violation.tool,
                    violation.detail.get("args", {}),
                    user=trace.user_text(),
                    turn=violation.turn or 0,
                )
                checked += 1
                if decision.allowed:
                    disagreements.append(
                        (rule.name, trace.trace_id, violation.tool, violation.kind)
                    )

    assert checked, "the comparison matched nothing -- is it still wired up?"
    assert not disagreements, (
        "CI flagged these calls but the runtime guard would have allowed them: "
        f"{disagreements}"
    )
def _intent_rule() -> Rule:
    return parse_rule(
        {
            "rule": "refund-safety",
            "when": {"intent_any": ["退款"]},
            "then": {"forbid_tool": ["refund_order"]},
        }
    )


def _write_rule(rule: Rule, path: Path) -> None:
    """`dump_rule` returns YAML text; tests need it on disk."""
    path.write_text(dump_rule(rule), encoding="utf-8")


def _two_turn_session() -> Trace:
    return Trace(
        trace_id="intent-drift",
        steps=(
            Step(turn=1, user="我想退款"),
            Step(
                turn=2,
                user="算了，帮我查下天气",
                tool_calls=(ToolCall(name="refund_order", turn=2),),
            ),
        ),
    )


def test_ci_flags_the_late_call_the_runtime_guard_used_to_miss():
    from agentguard.engine import evaluate

    kinds = [v.kind for v in evaluate(_intent_rule(), _two_turn_session()).violations]
    assert kinds == ["forbidden_tool"]


def test_runtime_history_closes_the_intent_bypass():
    guard = RuntimeGuard([_intent_rule()])

    # The current turn alone mentions no refund: the old behaviour, kept as a
    # guard against over-correcting (a caller with no history gets no history).
    assert guard.check("refund_order", {"order_id": "A1"}, user="算了，帮我查下天气", turn=2)

    decision = guard.check(
        "refund_order",
        {"order_id": "A1"},
        user="算了，帮我查下天气",
        turn=2,
        history=["我想退款"],
    )
    assert not decision.allowed
    assert decision.code == "forbidden_tool"


def test_runtime_history_defaults_to_the_old_single_turn_behaviour():
    """Adding `history` must not change any call that does not pass it."""
    guard = RuntimeGuard([_intent_rule()])
    without = guard.check("refund_order", {}, user="帮我退款", turn=1)
    with_empty = guard.check("refund_order", {}, user="帮我退款", turn=1, history=())
    assert (without.allowed, without.code) == (with_empty.allowed, with_empty.code)
    assert not without.allowed


def test_history_steps_get_sequential_turns():
    guard = RuntimeGuard([])
    trace = guard._context("当前", 0, ["第一轮", "第二轮"])
    assert [(s.turn, s.user) for s in trace.steps] == [
        (1, "第一轮"),
        (2, "第二轮"),
        (3, "当前"),
    ]
    # An explicit turn is honoured, not overwritten.
    trace = guard._context("当前", 9, ["第一轮"])
    assert [s.turn for s in trace.steps] == [1, 9]


def test_runtime_cannot_be_bypassed_by_dropping_the_trigger_word():
    """The safety half of the CI/runtime invariant, checked on real rules.

    Runtime legitimately knows less than CI -- it cannot see turns that have
    not happened yet.  So the invariant has to be directional: whenever CI
    flags a call at turn T *and* the rule's trigger words are already present
    in turns 1..T, the guard must refuse that call.  Anything else is a hole
    an agent can walk through.
    """
    from agentguard.engine import evaluate
    from agentguard.trace import load_traces

    repo = Path(__file__).resolve().parents[1]
    runtime_rules = [r for r in load_rules(repo / "rules") if r.is_runtime]
    assert runtime_rules, "expected runtime-enforced rules in rules/"
    guard = RuntimeGuard(runtime_rules)

    checked = 0
    holes = []
    for trace in load_traces(repo / "examples"):
        for rule in runtime_rules:
            for violation in evaluate(rule, trace).violations:
                if violation.kind not in ("forbidden_tool", "tool_not_allowed"):
                    continue
                if violation.tool is None or violation.turn is None:
                    continue
                earlier = [s.user for s in trace.steps if s.turn < violation.turn]
                current = trace.turn(violation.turn)
                if current is None:
                    continue
                prefix = "\n".join([*earlier, current.user]).lower()
                if rule.when.intent_any and not any(
                    kw.lower() in prefix for kw in rule.when.intent_any
                ):
                    # The trigger word is not known yet at this point in time,
                    # so runtime is allowed to differ from CI here.
                    continue

                decision = guard.check(
                    violation.tool,
                    violation.detail.get("args", {}),
                    user=current.user,
                    turn=violation.turn,
                    history=earlier,
                )
                checked += 1
                if decision.allowed:
                    holes.append(
                        (rule.name, trace.trace_id, violation.tool, violation.turn)
                    )

    assert checked, "the comparison matched nothing -- is it still wired up?"
    assert not holes, (
        "these calls are reachable at runtime even though CI flags them: "
        f"{holes}"
    )


# --------------------------------------------------------------------------- #
# FEATURE: async entry points.  Without them an async agent has to hand-roll
# check/record around every awaited tool -- and the copy that forgets `record`
# silently disables the idempotency check.
# --------------------------------------------------------------------------- #
def _idempotent_rule() -> Rule:
    return parse_rule(
        {
            "rule": "charge-once",
            "then": {"idempotent": {"tool": "refund_order", "key": "${order_id}"}},
        }
    )


def test_async_check_agrees_with_the_sync_check():
    import asyncio

    rule = _intent_rule()
    guard = RuntimeGuard([rule])
    sync = guard.check("refund_order", {}, user="退款", turn=1)
    asynced = asyncio.run(guard.acheck("refund_order", {}, user="退款", turn=1))
    assert (sync.allowed, sync.code) == (asynced.allowed, asynced.code)


def test_async_execute_awaits_records_and_blocks():
    import asyncio

    guard = RuntimeGuard([_idempotent_rule()])
    calls: list[str] = []

    async def tool(order_id: str) -> str:
        calls.append(order_id)
        return "ok"

    async def scenario() -> tuple[str, str]:
        first = await guard.aexecute("refund_order", {"order_id": "A1"}, lambda: tool("A1"))
        second = await guard.aexecute("refund_order", {"order_id": "A1"}, lambda: tool("A1"))
        return first, second

    from agentguard.executors.runtime import ToolBlocked

    with pytest.raises(ToolBlocked) as excinfo:
        asyncio.run(scenario())

    assert excinfo.value.decision.code == "duplicate_transaction"
    # The second call must never reach the tool -- and the first must have.
    assert calls == ["A1"]


def test_async_execute_does_not_await_the_tool_when_blocked():
    import asyncio

    guard = RuntimeGuard([_intent_rule()])
    ran = False

    async def tool() -> str:
        nonlocal ran
        ran = True
        return "ok"

    from agentguard.executors.runtime import ToolBlocked

    async def scenario() -> None:
        await guard.aexecute("refund_order", {}, lambda: tool(), user="退款", turn=1)

    with pytest.raises(ToolBlocked):
        asyncio.run(scenario())
    assert ran is False


def test_audit_trail_carries_the_decision_code():
    from agentguard.executors.runtime import ToolBlocked

    guard = RuntimeGuard([_intent_rule()])
    with pytest.raises(ToolBlocked):
        guard.execute("refund_order", {}, lambda: "nope", user="退款", turn=1)
    assert guard.audit[0]["code"] == "forbidden_tool"


# --------------------------------------------------------------------------- #
# FEATURE: SARIF output, so violations land in GitHub's Security tab.
# --------------------------------------------------------------------------- #
def _sarif_payload(outcome: GateOutcome) -> dict:
    from agentguard.report import render_sarif

    return json.loads(render_sarif(outcome.runs, outcome.rules))


def test_sarif_has_the_required_shape(tmp_path):
    rules_dir = tmp_path / "rules"
    traces_dir = tmp_path / "traces"
    rules_dir.mkdir()
    traces_dir.mkdir()
    _write_rule(_intent_rule(), rules_dir / "r.yaml")
    (traces_dir / "t.json").write_text(
        json.dumps(
            {
                "trace_id": "t",
                "steps": [
                    {"turn": 1, "user": "我想退款", "tool_calls": [{"name": "refund_order"}]}
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = _sarif_payload(run_gate(rules_dir, traces_dir))

    assert payload["version"] == "2.1.0"
    assert payload["$schema"].endswith("sarif-2.1.0.json")
    run = payload["runs"][0]
    assert run["tool"]["driver"]["name"] == "AgentGuard"
    assert run["tool"]["driver"]["version"]

    # Every ruleId used by a result must be declared in driver.rules, or
    # GitHub rejects the upload.
    declared = {r["id"] for r in run["tool"]["driver"]["rules"]}
    assert declared == {"refund-safety"}

    (result,) = run["results"]
    assert result["ruleId"] == "refund-safety"
    assert result["level"] == "error"
    assert "refund_order" in result["message"]["text"]
    # Machine-readable companion fields.
    assert result["properties"]["kind"] == "forbidden_tool"
    assert result["properties"]["severity"] == "critical"
    assert result["properties"]["turn"] == 1
    assert result["properties"]["canonicalMessage"]


def test_sarif_uri_points_into_the_repo_not_just_a_basename(tmp_path, monkeypatch):
    """`Trace.source` used to be `path.name`, so SARIF said `t.json` and GitHub
    could not resolve it to a file in the repository."""
    from agentguard.trace import load_trace

    monkeypatch.chdir(tmp_path)
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces" / "t.json").write_text(
        json.dumps({"trace_id": "t", "steps": [{"turn": 1, "user": "hi"}]}),
        encoding="utf-8",
    )
    assert load_trace(Path("traces/t.json")).source == "traces/t.json"


def test_sarif_levels_track_severity():
    from agentguard.report import _SARIF_LEVEL

    assert _SARIF_LEVEL == {"critical": "error", "warning": "warning", "info": "note"}


def test_sarif_is_written_into_the_report_bundle(tmp_path):
    rules_dir = tmp_path / "rules"
    traces_dir = tmp_path / "traces"
    rules_dir.mkdir()
    traces_dir.mkdir()
    _write_rule(_intent_rule(), rules_dir / "r.yaml")
    (traces_dir / "t.json").write_text(
        json.dumps(
            {
                "trace_id": "t",
                "steps": [
                    {"turn": 1, "user": "我想退款", "tool_calls": [{"name": "refund_order"}]}
                ],
            }
        ),
        encoding="utf-8",
    )
    from agentguard.executors.ci import write_reports

    paths = write_reports(run_gate(rules_dir, traces_dir), tmp_path / "out")
    assert set(paths) == {"json", "markdown", "sarif"}
    assert json.loads(paths["sarif"].read_text(encoding="utf-8"))["version"] == "2.1.0"


def test_cli_gate_emits_parseable_sarif(tmp_path, capsys):
    from agentguard.cli import main

    rules_dir = tmp_path / "rules"
    traces_dir = tmp_path / "traces"
    rules_dir.mkdir()
    traces_dir.mkdir()
    _write_rule(_intent_rule(), rules_dir / "r.yaml")
    (traces_dir / "t.json").write_text(
        json.dumps(
            {
                "trace_id": "t",
                "steps": [
                    {"turn": 1, "user": "我想退款", "tool_calls": [{"name": "refund_order"}]}
                ],
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "gate",
            "--rules",
            str(rules_dir),
            "--traces",
            str(traces_dir),
            "--format",
            "sarif",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["runs"][0]["results"][0]["ruleId"] == "refund-safety"


# --------------------------------------------------------------------------- #
# KNOWN WART, pinned so it cannot change silently: `when.turn_range` gates
# whether a rule *activates*, not which calls it inspects.  A rule "scoped to
# turn 1" therefore still flags a forbidden tool at turn 9.  That is what CI
# has always done; the test records it rather than pretending otherwise.
# --------------------------------------------------------------------------- #
def test_turn_range_gates_activation_not_scope():
    from agentguard.engine import evaluate

    rule = parse_rule(
        {
            "rule": "turn-scoped",
            "when": {"turn_range": [1, 1]},
            "then": {"forbid_tool": ["refund_order"]},
        }
    )
    trace = Trace(
        trace_id="t",
        steps=(
            Step(turn=1, user="你好"),
            Step(
                turn=9,
                user="再见",
                tool_calls=(ToolCall(name="refund_order", turn=9),),
            ),
        ),
    )
    violations = evaluate(rule, trace).violations
    assert [(v.tool, v.turn) for v in violations] == [("refund_order", 9)]


def test_sarif_rule_descriptions_are_stripped(tmp_path):
    """YAML block scalars carry a trailing newline; it looked like a typo in
    the GitHub alert help panel."""
    rules_dir = tmp_path / "rules"
    traces_dir = tmp_path / "traces"
    rules_dir.mkdir()
    traces_dir.mkdir()
    (rules_dir / "r.yaml").write_text(
        "version: 1\n"
        "rule: refund-safety\n"
        "description: |\n"
        "  只读场景下不得调用破坏性工具。\n"
        "when:\n"
        '  intent_any: ["退款"]\n'
        "then:\n"
        "  forbid_tool: refund_order\n",
        encoding="utf-8",
    )
    (traces_dir / "t.json").write_text(
        json.dumps(
            {
                "trace_id": "t",
                "steps": [
                    {"turn": 1, "user": "我想退款", "tool_calls": [{"name": "refund_order"}]}
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = _sarif_payload(run_gate(rules_dir, traces_dir))
    (driver_rule,) = payload["runs"][0]["tool"]["driver"]["rules"]
    assert driver_rule["shortDescription"]["text"] == "只读场景下不得调用破坏性工具。"


def test_guard_cli_warns_when_an_intent_rule_is_checked_without_history(tmp_path, capsys):
    """An ALLOW from an intent-gated rule with no history is not a real
    clearance -- CI would have blocked.  It must not be reported as a silent
    green."""
    from agentguard.cli import main

    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    _write_rule(_intent_rule(), rules_dir / "r.yaml")

    code = main(
        [
            "guard",
            "--rules",
            str(rules_dir),
            "--tool",
            "refund_order",
            "--user",
            "算了，帮我查下天气",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0  # allowed -- but loudly qualified
    assert "--history" in out

    # With history supplied the caveat must disappear.
    code = main(
        [
            "guard",
            "--rules",
            str(rules_dir),
            "--tool",
            "refund_order",
            "--user",
            "算了，帮我查下天气",
            "--history",
            "我想退款",
        ]
    )
    out = capsys.readouterr().out
    assert code == 2
    assert "BLOCK" in out


def test_public_surface_matches___all__():
    """`__all__` is documentation; if it drifts from reality, `from agentguard
    import X` fails for a user instead of for us."""
    import agentguard

    missing = sorted(n for n in agentguard.__all__ if not hasattr(agentguard, n))
    assert not missing, f"__all__ lists names that do not exist: {missing}"


def test_ledger_error_is_reachable_from_the_top_level_package(tmp_path):
    """Callers must be able to `except agentguard.LedgerError` without knowing
    which submodule it lives in -- it is raised from `FileStore.__init__`."""
    from agentguard import FileStore, LedgerError

    ledger = tmp_path / "ledger.json"
    ledger.write_text("{ not json", encoding="utf-8")
    with pytest.raises(LedgerError):
        FileStore(ledger)


def test_sarif_declares_the_ruleset_even_on_a_clean_run(tmp_path):
    """`results: []` + `rules: []` reads as "nothing ran".  A clean gate should
    still show which ruleset was applied."""
    rules_dir = tmp_path / "rules"
    traces_dir = tmp_path / "traces"
    rules_dir.mkdir()
    traces_dir.mkdir()
    _write_rule(_intent_rule(), rules_dir / "r.yaml")
    (traces_dir / "t.json").write_text(
        json.dumps({"trace_id": "t", "steps": [{"turn": 1, "user": "今天天气不错"}]}),
        encoding="utf-8",
    )

    payload = _sarif_payload(run_gate(rules_dir, traces_dir))
    run = payload["runs"][0]
    assert run["results"] == []
    assert [r["id"] for r in run["tool"]["driver"]["rules"]] == ["refund-safety"]
def test_forbid_and_allow_do_not_double_report_the_same_call():
    from agentguard.engine import evaluate

    rule = parse_rule(
        {
            "rule": "redundant",
            "then": {"forbid_tool": ["refund_order"], "allow_tool": ["query_order"]},
        }
    )
    trace = Trace(
        trace_id="t",
        steps=(
            Step(
                turn=3,
                user="帮我查物流",
                tool_calls=(ToolCall(name="refund_order", turn=3),),
            ),
        ),
    )
    violations = evaluate(rule, trace).violations
    assert [(v.kind, v.tool, v.turn) for v in violations] == [
        ("forbidden_tool", "refund_order", 3)
    ]


def test_dedupe_keeps_genuinely_distinct_findings():
    """Deduping must not swallow real duplicates: two transactions at different
    turns are two findings, and different tools are different findings."""
    from agentguard.engine import evaluate

    rule = parse_rule(
        {
            "rule": "charge-once",
            "then": {
                "forbid_tool": ["refund_order"],
                "idempotent": {"tool": "refund_order", "key": "${order_id}"},
            },
        }
    )
    trace = Trace(
        trace_id="t",
        steps=(
            Step(turn=1, user="a", tool_calls=(ToolCall("refund_order", {"order_id": "A"}, 1),)),
            Step(turn=2, user="b", tool_calls=(ToolCall("refund_order", {"order_id": "A"}, 2),)),
        ),
    )
    kinds = [(v.kind, v.turn) for v in evaluate(rule, trace).violations]
    # Both calls are forbidden (two turns) and the second is a duplicate.
    assert ("forbidden_tool", 1) in kinds
    assert ("forbidden_tool", 2) in kinds
    assert ("duplicate_transaction", 2) in kinds
    assert len(kinds) == len(set(kinds))


def test_dedupe_does_not_merge_two_different_tools():
    from agentguard.engine import evaluate

    rule = parse_rule(
        {
            "rule": "no-money-tools",
            "then": {"forbid_tool": ["refund_order", "charge_card"]},
        }
    )
    trace = Trace(
        trace_id="t",
        steps=(
            Step(
                turn=1,
                user="x",
                tool_calls=(
                    ToolCall(name="refund_order", turn=1),
                    ToolCall(name="charge_card", turn=1),
                ),
            ),
        ),
    )
    tools = sorted(v.tool for v in evaluate(rule, trace).violations)
    assert tools == ["charge_card", "refund_order"]


def test_learned_rule_reports_each_bad_call_once(tmp_path):
    """End-to-end: `learn --apply` then `run` must not double-count."""
    from agentguard.learn import learn_from_trace
    from agentguard.trace import parse_trace

    trace = parse_trace(
        {
            "trace_id": "incident",
            "steps": [
                {
                    "turn": 3,
                    "user": "帮我查下物流",
                    "tool_calls": [
                        {"name": "query_order", "args": {"order_id": "A"}},
                        {"name": "refund_order", "args": {"order_id": "A"}},
                    ],
                }
            ],
        }
    )
    learned = [c.rule for c in learn_from_trace(trace) if "forbid" in c.rule.name]
    assert learned, "expected a forbid candidate"

    from agentguard.engine import evaluate

    result = evaluate(learned[0], trace)
    assert len(result.violations) == 1, [v.kind for v in result.violations]


# --------------------------------------------------------------------------- #
# BUG 26: `guard` printed a clean "[ALLOW]" for a tool no rule had ever heard
# of -- which is also exactly what a typo in `--tool` looks like.  "I have
# nothing to say about this tool" is not "this tool is safe".
# --------------------------------------------------------------------------- #
def test_is_policed_distinguishes_unmodelled_tools_from_known_ones():
    rule = parse_rule({"rule": "r", "then": {"forbid_tool": ["refund_order"]}})
    guard = RuntimeGuard([rule])
    assert guard.is_policed("refund_order")
    assert not guard.is_policed("refund_orders")  # typo
    assert not guard.is_policed("nonexistent_tool")


def test_guard_cli_says_when_no_rule_mentions_the_tool(tmp_path, capsys):
    from agentguard.cli import main

    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    _write_rule(
        parse_rule({"rule": "r", "then": {"forbid_tool": ["refund_order"]}}),
        rules_dir / "r.yaml",
    )

    code = main(["guard", "--rules", str(rules_dir), "--tool", "refund_orderz", "--user", "hi"])
    out = capsys.readouterr().out
    assert code == 0
    assert "无话可说" in out
    assert "refund_order" in out  # tells you what IS policed


def test_is_policed_ignores_rules_that_never_run():
    """A CI-only rule does not police anything at runtime."""
    rule = parse_rule(
        {
            "rule": "ci-only",
            "then": {"forbid_tool": ["refund_order"]},
            "where": {"ci": True, "runtime": False},
        }
    )
    guard = RuntimeGuard([rule])
    assert not guard.is_policed("refund_order")
def test_shipped_rules_stop_a_duplicate_refund_with_no_trigger_word():
    from agentguard.idempotency import MemoryStore, FileStore

    repo = Path(__file__).resolve().parents[1]
    guard = RuntimeGuard.from_rules(repo / "rules", store=MemoryStore())

    for _ in range(2):
        decision = guard.check(
            "refund_order", {"order_id": "A1001"}, user="那我再等等", turn=3
        )
        if decision.allowed:
            guard.record("refund_order", {"order_id": "A1001"})

    assert not decision.allowed, (
        "the second refund of the same order was allowed -- idempotency is "
        "being masked by the `when` gate again"
    )
    assert decision.code == "duplicate_transaction"


def test_no_shipped_rule_hides_idempotency_behind_a_when_gate():
    """Structural guard: the shape that caused BUG 27 must not come back."""
    repo = Path(__file__).resolve().parents[1]
    offenders = [
        rule.name
        for rule in load_rules(repo / "rules")
        if rule.then.idempotent and not rule.when.always
    ]
    assert not offenders, (
        f"these rules gate `idempotent` behind `when`, so the check silently "
        f"stops running on turns that do not match: {offenders}"
    )


def test_warns_when_idempotency_is_gated_by_when():
    """If a user does write that shape, they must be told -- not left with a
    guard that quietly does nothing."""
    rule = parse_rule(
        {
            "rule": "gated",
            "when": {"intent_any": ["退款"]},
            "then": {"idempotent": {"tool": "charge_card", "key": "${order_id}"}},
        }
    )
    warnings = validate_rules([rule])
    assert any("idempotent" in w and "when" in w for w in warnings), warnings


def test_ungated_idempotency_does_not_warn():
    rule = parse_rule(
        {"rule": "ok", "then": {"idempotent": {"tool": "charge_card", "key": "${order_id}"}}}
    )
    assert not [w for w in validate_rules([rule]) if "idempotent" in w]


# --------------------------------------------------------------------------- #
# BUG 28: eleven tests were defined twice in this very file.
#
# They had been appended by `cat >>` across several turns.  Python keeps the
# *last* definition and silently discards the earlier ones, so those tests
# never ran -- the count kept growing while coverage did not.  One of the
# buried copies had also been corrupted (`"when:/n"` instead of `"when:/n"`,
# courtesy of MSYS mangling the escape), which is why the damage was invisible.
# --------------------------------------------------------------------------- #
def test_no_test_function_is_defined_twice():
    import ast
    import collections

    repo = Path(__file__).resolve().parents[1]
    offenders: dict[str, list[str]] = {}
    for path in sorted((repo / "tests").rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        ]
        dupes = [n for n, c in collections.Counter(names).items() if c > 1]
        if dupes:
            offenders[path.name] = dupes

    assert not offenders, (
        "these test functions are defined more than once, so the earlier "
        f"definitions never execute: {offenders}"
    )


# --------------------------------------------------------------------------- #
# BUG 29: the terminal and Markdown reports lost the trace id.
#
# `render_sarif` had always carried `properties.traceId`, so a SARIF alert
# could be traced back to its case.  The two human-facing renderers did not:
# with three traces in `examples/`, `[BLOCK] refund-safety` appeared twice and
# a reader had no way to tell which case each line came from.  The same data
# was already in hand -- only the rendering dropped it.
# --------------------------------------------------------------------------- #
def _two_traces_same_rule() -> list[tuple[Trace, list]]:
    """Two traces that both violate one rule, so the ids must distinguish them."""
    from agentguard.engine import evaluate

    rule = parse_rule(
        {
            "rule": "refund-safety",
            "then": {"forbid_tool": ["refund_order"]},
            "where": {"ci": True, "runtime": True},
        }
    )
    traces = [
        parse_trace(
            {
                "trace_id": trace_id,
                "steps": [
                    {
                        "turn": 1,
                        "user": "查一下",
                        "tool_calls": [{"name": "refund_order", "args": {}}],
                    }
                ],
            }
        )
        for trace_id in ("case-alpha", "case-beta")
    ]
    return [(t, [evaluate(rule, t)]) for t in traces]


def test_terminal_report_names_the_trace_for_each_violation():
    from agentguard.report import render_terminal

    text = render_terminal(_two_traces_same_rule())
    assert "case-alpha" in text and "case-beta" in text, text
    # The count must appear under each trace, not once for the whole run.
    assert text.count("refund-safety") == 2, text


def test_markdown_report_has_a_trace_column():
    from agentguard.report import render_markdown

    text = render_markdown(_two_traces_same_rule())
    assert "| Trace |" in text, text
    assert "case-alpha" in text and "case-beta" in text, text


# --------------------------------------------------------------------------- #
# BUG 30/31: `rules/refund-safety.yaml` allowed only English tool names, so
# `examples/refund_zh` (Chinese tool names, same Chinese user utterances) was
# judged against a list it could never satisfy -- four violations whose stated
# cause ("name not in the allow-list") had nothing to do with what the rule
# meant ("do not refund in a read-only scenario").
#
# Splitting it into one rule per language does NOT work, which is worth a test:
# `when.intent_any` matches the *user's words*, and both traces speak Chinese,
# so each rule's `when` fires on the other's trace and the two allow-lists
# mis-judge each other in both directions.  The fix is to drop `allow_tool`.
# --------------------------------------------------------------------------- #
def test_shipped_refund_rule_does_not_trap_other_naming_systems():
    """No shipped rule may pair `forbid_tool` with an `allow_tool` that cannot
    cover it -- that shape is what produced the false positives."""
    shipped = load_rules(Path(__file__).resolve().parents[1] / "rules")
    offenders = [
        r.name
        for r in shipped
        if r.then.forbid_tool
        and r.then.allow_tool
        and set(r.then.forbid_tool) - set(r.then.allow_tool)
    ]
    assert not offenders, (
        f"{offenders} forbid a tool that its own allow-list already excludes; "
        "the allow-list is a closed list and will false-positive on any other "
        "tool-naming scheme"
    )


def test_allow_list_absent_means_other_languages_are_not_flagged():
    """The exact regression: a Chinese tool call must not be reported as
    "outside the allow-list" by a rule that never named Chinese tools."""
    from agentguard.engine import evaluate

    rule = parse_rule(
        {
            "rule": "refund-safety",
            "when": {"intent_any": ["物流"]},
            "then": {"forbid_tool": ["refund_order"]},
            "where": {"ci": True, "runtime": True},
        }
    )
    trace = parse_trace(
        {
            "trace_id": "退款事故",
            "steps": [
                {
                    "turn": 1,
                    "user": "帮我看下物流",
                    "tool_calls": [{"name": "查询订单", "args": {"订单号": "CN-1"}}],
                }
            ],
        }
    )
    result = evaluate(rule, trace)
    assert result.violations == (), [v.message for v in result.violations]


def test_chinese_trace_is_still_policed_by_the_idempotency_rule():
    """Dropping the allow-list must not make the Chinese case unguarded."""
    from agentguard.executors.ci import run_gate

    root = Path(__file__).resolve().parents[1]
    outcome = run_gate(root / "rules", root / "examples", fail_on="critical")
    zh = [
        (t, v)
        for t, results in outcome.runs
        for r in results
        for v in r.violations
        if t.trace_id == "退款事故-中文工具名"
    ]
    assert zh, "the Chinese trace must still trip at least one rule"
    assert all(v.kind == "duplicate_transaction" for _t, v in zh), [
        (t.trace_id, v.kind) for t, v in zh
    ]


# --------------------------------------------------------------------------- #
# BUG 32: `validate_rules` never looked at `forbid_tool` + `allow_tool` in the
# same rule -- the very shape `learn.py` documents 12 lines of comment about
# avoiding.  The learner was protected; hand-written rules were not.
# --------------------------------------------------------------------------- #
def test_warns_when_forbid_is_subsumed_by_the_allow_list():
    rule = parse_rule(
        {
            "rule": "subsumed",
            "then": {
                "forbid_tool": ["refund_order"],
                "allow_tool": ["query_order"],
            },
        }
    )
    warnings = validate_rules([rule])
    assert any("allow_tool" in w and "forbid_tool" in w for w in warnings), warnings


def test_warns_when_a_tool_is_both_allowed_and_forbidden():
    rule = parse_rule(
        {
            "rule": "contradiction",
            "then": {"forbid_tool": ["refund_order"], "allow_tool": ["refund_order"]},
        }
    )
    warnings = validate_rules([rule])
    assert any("既允许又禁止" in w for w in warnings), warnings


def test_forbid_alone_does_not_warn():
    rule = parse_rule({"rule": "fine", "then": {"forbid_tool": ["refund_order"]}})
    assert not [w for w in validate_rules([rule]) if "allow_tool" in w]


def test_allow_list_that_includes_the_forbidden_tool_is_still_flagged():
    """`allow: [a, b]` + `forbid: [a]` leaves `a` both permitted and refused."""
    rule = parse_rule(
        {
            "rule": "mixed",
            "then": {"forbid_tool": ["refund_order"], "allow_tool": ["refund_order", "query_order"]},
        }
    )
    warnings = validate_rules([rule])
    assert any("refund_order" in w for w in warnings), warnings


# --------------------------------------------------------------------------- #
# BUG 35: the `--history` warning named rules that had nothing to do with the
# call.  It listed every `when`-gated rule, so a `charge_card` check warned
# about `refund-safety` -- a rule whose `forbid_tool` is `refund_order`.
# A warning that names an irrelevant rule teaches people to ignore warnings.
# --------------------------------------------------------------------------- #
def test_history_warning_only_names_rules_that_cover_the_tool(tmp_path, capsys):
    from agentguard.cli import main

    (tmp_path / "rules").mkdir()
    _write_rule(
        parse_rule(
            {
                "rule": "refund-safety",
                "when": {"intent_any": ["物流"]},
                "then": {"forbid_tool": ["refund_order"]},
                "where": {"ci": True, "runtime": True},
                "severity": "critical",
            }
        ),
        tmp_path / "rules" / "refund-safety.yaml",
    )

    # charge_card is not covered by refund-safety -> no intent warning.
    code = main(
        [
            "guard",
            "--rules",
            str(tmp_path / "rules"),
            "--tool",
            "charge_card",
            "--arg",
            "order_id=A1",
            "--user",
            "我要付款",
        ]
    )
    out = capsys.readouterr().out
    assert "refund-safety" not in out, out

    # refund_order *is* covered -> the warning must appear.
    main(
        [
            "guard",
            "--rules",
            str(tmp_path / "rules"),
            "--tool",
            "refund_order",
            "--arg",
            "order_id=A1",
            "--user",
            "随便说点别的",
        ]
    )
    out = capsys.readouterr().out
    assert "refund-safety" in out and "--history" in out, out


# --------------------------------------------------------------------------- #
# BUG 36: a tool call whose name could not be read was silently dropped.
#
# `_coerce_tool_call` returned None for `{"args": {}}`, so a trace whose
# tool-name field is spelled differently (or nested in a shape we do not know)
# reported `tool_calls=0` and a green PASS.  "I could not read this" and "there
# was nothing here" must not look the same -- that is the exact silent failure
# AgentGuard exists to prevent.
# --------------------------------------------------------------------------- #
def test_unreadable_tool_call_is_counted_not_dropped():
    trace = parse_trace(
        {
            "trace_id": "opaque",
            "steps": [
                {
                    "turn": 1,
                    "user": "x",
                    "tool_calls": [{"args": {"order_id": "A"}}],
                }
            ],
        }
    )
    from agentguard.trace import UNNAMED_TOOL

    assert len(trace.tool_calls()) == 1, "the call must not vanish"
    assert trace.tool_calls()[0].name == UNNAMED_TOOL


def test_unreadable_tool_call_produces_a_load_warning():
    from agentguard.trace import WARNINGS_KEY

    trace = parse_trace(
        {
            "trace_id": "opaque",
            "steps": [{"turn": 1, "user": "x", "tool_calls": [{"args": {}}]}],
        }
    )
    warnings = trace.metadata[WARNINGS_KEY]
    assert warnings, trace.metadata
    assert any("读不到名字" in w for w in warnings), warnings


def test_unnamed_call_cannot_trigger_a_rule():
    """The placeholder must not match anything -- a fix that traded silence for
    a false positive would be worse."""
    from agentguard.engine import evaluate

    rule = parse_rule({"rule": "r", "then": {"forbid_tool": ["refund_order"]}})
    trace = parse_trace(
        {
            "trace_id": "opaque",
            "steps": [{"turn": 1, "user": "x", "tool_calls": [{"args": {}}]}],
        }
    )
    assert evaluate(rule, trace).violations == ()


def test_readable_tool_calls_produce_no_warning():
    from agentguard.trace import WARNINGS_KEY

    trace = parse_trace(
        {
            "trace_id": "fine",
            "steps": [
                {
                    "turn": 1,
                    "user": "x",
                    "tool_calls": [{"name": "query_order", "args": {}}],
                }
            ],
        }
    )
    assert WARNINGS_KEY not in trace.metadata


def test_terminal_report_shows_the_trace_warning():
    """stderr is not enough -- CI often keeps stdout only."""
    from agentguard.engine import evaluate
    from agentguard.report import render_terminal

    rule = parse_rule({"rule": "r", "then": {"forbid_tool": ["refund_order"]}})
    trace = parse_trace(
        {
            "trace_id": "opaque",
            "steps": [{"turn": 1, "user": "x", "tool_calls": [{"args": {}}]}],
        }
    )
    text = render_terminal([(trace, [evaluate(rule, trace)])])
    assert "[!]" in text and "读不到名字" in text, text


# --------------------------------------------------------------------------- #
# BUG 37: one malformed rule file aborted the whole directory load, so the user
# saw only the first error and had to fix-rerun for each one.  Still fail-fast
# (a skipped rule is indistinguishable from a passing rule), but now every
# file's error is reported in one pass.
# --------------------------------------------------------------------------- #
def test_all_rule_errors_are_reported_together(tmp_path):
    from agentguard.rules import load_rules

    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    # Written as raw text: `parse_rule` would raise before we could save them.
    (rules_dir / "a.yaml").write_text(
        "version: 1\nrule: a\nthen:\n  forbid_tool: [x]\n  typo: 1\n",
        encoding="utf-8",
    )
    (rules_dir / "b.yaml").write_text(
        "version: 1\nthen:\n  forbid_tool: [y]\n", encoding="utf-8"
    )

    with pytest.raises(RuleError) as exc:
        load_rules(rules_dir)

    message = str(exc.value)
    assert "2 条规则无法加载" in message, message
    assert "typo" in message, message
    assert "rule:" in message, message


def test_a_single_bad_rule_still_refuses_to_load(tmp_path):
    """Aggregating errors must not turn into skipping them."""
    from agentguard.rules import load_rules

    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    _write_rule(
        parse_rule({"rule": "good", "then": {"forbid_tool": ["x"]}}),
        rules_dir / "g.yaml",
    )
    (rules_dir / "b.yaml").write_text(
        "version: 1\nrule: bad\nthen:\n  nope: 1\n", encoding="utf-8"
    )

    with pytest.raises(RuleError):
        load_rules(rules_dir)


def test_healthy_rules_still_load_without_error(tmp_path):
    from agentguard.rules import load_rules

    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    _write_rule(
        parse_rule({"rule": "ok", "then": {"forbid_tool": ["x"]}}),
        rules_dir / "ok.yaml",
    )
    assert [r.name for r in load_rules(rules_dir)] == ["ok"]


# --------------------------------------------------------------------------- #
# BUG 39: the Chinese renderers were reachable only via
# `agentguard.messages`, and the README never mentioned them.  A user who
# followed the integration guide got English-only log lines and had no way to
# know a Chinese option existed -- even though localisation is a headline
# feature.  Both are exported from the top level now, and the README shows the
# call.  This test pins the public entry point, not the implementation.
# --------------------------------------------------------------------------- #
def test_chinese_renderers_are_exported_from_the_package_root():
    import agentguard

    for name in ("decision_zh", "violation_zh"):
        assert name in agentguard.__all__, f"{name} missing from __all__"
        assert callable(getattr(agentguard, name)), name


def test_decision_zh_renders_a_runtime_denial():
    from agentguard import decision_zh

    text = decision_zh(
        code="forbidden_tool",
        reason="tool 'refund_order' is forbidden by rule 'refund-safety'",
        detail={"tool": "refund_order"},
        rule="refund-safety",
    )
    assert "refund-safety" in text
    assert "禁止" in text


def test_decision_zh_falls_back_to_the_canonical_reason():
    """An unmapped code must not render as an empty string."""
    from agentguard import decision_zh

    assert decision_zh(code="something_new", reason="canonical text") == "canonical text"


def test_violation_zh_no_longer_wraps_the_message_for_known_kinds():
    from agentguard import violation_zh

    text = violation_zh(
        kind="duplicate_transaction",
        message="duplicate transaction: 'x' already executed",
        detail={"key_template": "${order_id}", "key_value": "A1", "first_turn": 1},
        turn=2,
        tool="charge_card",
    )
    assert "重复事务" in text


# --------------------------------------------------------------------------- #
# GAP 1: the idempotency guard was completely absent under concurrency.
#
# `check` read the ledger, `fn` ran (in production: a payment call taking
# hundreds of milliseconds), and only then did `record` write to it.  Every
# caller that arrived inside that window saw an empty ledger.  Measured on the
# bundled rule with 12 simultaneous calls on one order id: 12 executions, 0
# blocks.  A guard that advertises idempotency and provides none is worse than
# no guard, because the operator stops watching for the duplicate.
#
# The fix claims the key *before* `fn` runs and hands it back if `fn` raises.
# --------------------------------------------------------------------------- #
_IDEM_ONLY = parse_rule(
    {
        "rule": "pay-idem",
        "then": {"idempotent": {"tool": "charge_card", "key": "${order_id}"}},
    }
)


def test_concurrent_calls_on_one_key_execute_exactly_once():
    import threading

    from agentguard.idempotency import MemoryStore

    guard = RuntimeGuard([_IDEM_ONLY], store=MemoryStore())
    executed: list[int] = []
    blocked: list[str] = []
    start = threading.Barrier(12)

    def worker() -> None:
        start.wait()
        try:
            guard.execute("charge_card", {"order_id": "C1"}, lambda: executed.append(1) or "ok")
        except ToolBlocked as exc:
            blocked.append(exc.decision.code)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(executed) == 1, (
        f"{len(executed)} charges went through for one order id -- the "
        f"reservation is not happening before `fn` runs"
    )
    assert len(blocked) == 12 - 1
    assert set(blocked) == {"duplicate_transaction"}


def test_concurrent_calls_on_distinct_keys_all_execute():
    """The reservation must not serialise unrelated transactions."""
    import threading

    from agentguard.idempotency import MemoryStore

    guard = RuntimeGuard([_IDEM_ONLY], store=MemoryStore())
    executed: list[str] = []
    start = threading.Barrier(8)

    def worker(order: str) -> None:
        start.wait()
        guard.execute("charge_card", {"order_id": order}, lambda: executed.append(order) or "ok")

    threads = [threading.Thread(target=worker, args=(f"O{i}",)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(executed) == [f"O{i}" for i in range(8)]


def test_failed_tool_call_releases_its_reservation():
    """A charge that raised did not move money, so retrying it is not a
    duplicate -- refusing the retry would strand the order forever."""
    from agentguard.idempotency import MemoryStore

    store = MemoryStore()
    guard = RuntimeGuard([_IDEM_ONLY], store=store)

    def boom() -> None:
        raise RuntimeError("gateway timeout")

    with pytest.raises(RuntimeError, match="gateway timeout"):
        guard.execute("charge_card", {"order_id": "R1"}, boom)

    assert store.as_dict() == {}, "the claim leaked, so no retry can ever succeed"

    assert guard.execute("charge_card", {"order_id": "R1"}, lambda: "ok") == "ok"
    with pytest.raises(ToolBlocked):
        guard.execute("charge_card", {"order_id": "R1"}, lambda: "ok")


def test_lost_race_does_not_leak_a_claim():
    """When two keys are policed and the second one is taken, the first must be
    handed back -- otherwise a partial claim blocks a legitimate future call."""
    from agentguard.idempotency import MemoryStore

    two_keys = parse_rule(
        {
            "rule": "two",
            "then": {
                "idempotent": [
                    {"tool": "transfer", "key": "${from_id}"},
                    {"tool": "transfer", "key": "${to_id}"},
                ]
            },
        }
    )
    store = MemoryStore()
    store.record("two:transfer:'B'")  # the second key is already burned
    guard = RuntimeGuard([two_keys], store=store)

    with pytest.raises(ToolBlocked):
        guard.execute("transfer", {"from_id": "A", "to_id": "B"}, lambda: "ok")

    assert not store.seen("two:transfer:'A'"), (
        "the first key stayed claimed after the second one lost, so a later "
        "legitimate transfer from A would be refused as a duplicate"
    )


def test_file_store_reservation_is_atomic(tmp_path):
    import threading

    from agentguard.idempotency import FileStore

    store = FileStore(tmp_path / "ledger.json")
    start = threading.Barrier(16)
    winners: list[int] = []

    def worker() -> None:
        start.wait()
        if store.reserve("k"):
            winners.append(1)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(winners) == 1, f"{len(winners)} callers reserved the same key"
    assert FileStore(tmp_path / "ledger.json").seen("k")


# --------------------------------------------------------------------------- #
# GAP 2: the new reservation API broke every custom store.
#
# `_reserve` called `self.store.reserve(...)` unconditionally, so an existing
# integration -- any store implementing only `seen`/`record`, which is all the
# documented interface ever required -- crashed with an AttributeError on the
# first guarded call.  Breaking a working integration is a worse outcome than
# running without the concurrency fix, so the guard now degrades (and says so).
# --------------------------------------------------------------------------- #
class _LegacyStore:
    """The pre-`reserve` interface, verbatim."""

    def __init__(self) -> None:
        self._keys: set[str] = set()

    def seen(self, key: str) -> bool:
        return key in self._keys

    def record(self, key: str) -> None:
        self._keys.add(key)


def test_legacy_store_does_not_crash():
    guard = RuntimeGuard([_IDEM_ONLY], store=_LegacyStore())
    assert guard.execute("charge_card", {"order_id": "L1"}, lambda: "ok") == "ok"


def test_legacy_store_still_blocks_a_sequential_duplicate():
    guard = RuntimeGuard([_IDEM_ONLY], store=_LegacyStore())
    guard.execute("charge_card", {"order_id": "L2"}, lambda: "ok")
    with pytest.raises(ToolBlocked) as caught:
        guard.execute("charge_card", {"order_id": "L2"}, lambda: "ok")
    assert caught.value.decision.code == "duplicate_transaction"


def test_legacy_store_rolls_back_a_failed_call():
    store = _LegacyStore()
    guard = RuntimeGuard([_IDEM_ONLY], store=store)

    with pytest.raises(RuntimeError):
        guard.execute("charge_card", {"order_id": "L3"}, lambda: (_ for _ in ()).throw(RuntimeError()))
    assert store._keys == set()
    assert guard.execute("charge_card", {"order_id": "L3"}, lambda: "ok") == "ok"


def test_legacy_store_is_reported_as_a_degradation():
    """Silently losing a concurrency guarantee is the failure mode this project
    exists to eliminate -- the user must be told."""
    guard = RuntimeGuard([_IDEM_ONLY], store=_LegacyStore())
    assert any("reserve" in w for w in guard.warnings), guard.warnings


def test_atomic_store_produces_no_degradation_warning():
    from agentguard.idempotency import MemoryStore

    assert not RuntimeGuard([_IDEM_ONLY], store=MemoryStore()).warnings


def test_legacy_store_without_idempotency_rules_is_not_warned_about():
    """No idempotency rule means the store is never consulted, so there is
    nothing to degrade -- warning anyway is noise that trains users to ignore
    warnings."""
    forbid_only = parse_rule({"rule": "f", "then": {"forbid_tool": "refund_order"}})
    guard = RuntimeGuard([forbid_only], store=_LegacyStore())
    assert not [w for w in guard.warnings if "reserve" in w]


# --------------------------------------------------------------------------- #
# BUG 40: `learn` re-derived rules that already existed, and nothing said so.
#
# The documented flow is "derive -> review -> commit", and the natural moment
# to run it is right after an incident -- by which time hand-written rules
# covering the same ground usually exist.  Committing the learner's output on
# the bundled incident took the gate from 3 violations to 6 for one duplicate
# charge: still correct, but "violations" stopped meaning "problems", and any
# baseline recorded beforehand became meaningless.
# --------------------------------------------------------------------------- #
def _repo_with_rules(tmp_path: Path) -> Path:
    """A rules/ directory holding the shipped rules, for learn-overlap tests."""
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    for source in (Path(__file__).resolve().parents[1] / "rules").glob("*.yaml"):
        (rules_dir / source.name).write_text(
            source.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return rules_dir


def _learned_duplicates(rules_dir: Path) -> list[str]:
    """Run the same overlap detection the CLI uses."""
    existing = load_rules(rules_dir)
    covered = {}
    for rule in existing:
        for tool in rule.then.forbid_tool:
            covered.setdefault(("forbid_tool", tool), rule.name)
        for spec in rule.then.idempotent:
            covered.setdefault(("idempotent", f"{spec.tool}|{spec.key}"), rule.name)

    from agentguard.learn import learn_from_trace
    from agentguard.trace import load_trace

    trace = load_trace(Path(__file__).resolve().parents[1] / "examples/refund_incident/trace.json")
    hits = []
    for candidate in learn_from_trace(trace):
        rule = candidate.rule
        for tool in rule.then.forbid_tool:
            if ("forbid_tool", tool) in covered:
                hits.append(tool)
        for spec in rule.then.idempotent:
            if ("idempotent", f"{spec.tool}|{spec.key}") in covered:
                hits.append(spec.tool)
    return hits


def test_learn_detects_rules_that_already_exist(tmp_path):
    """The learner will re-derive what is already written by hand -- that is
    expected.  What matters is that the overlap is *detected*, so the user is
    not silently doubling their gate."""
    rules_dir = _repo_with_rules(tmp_path)
    duplicates = _learned_duplicates(rules_dir)
    assert duplicates, (
        "the learner re-derived the hand-written rules but no overlap was "
        "detected -- committing this doubles the violation count"
    )
    assert set(duplicates) == {"refund_order"}


def test_validate_rules_flags_a_duplicated_forbid(tmp_path):
    rules_dir = _repo_with_rules(tmp_path)
    extra = parse_rule(
        {
            "rule": "duplicate-forbid",
            "then": {"forbid_tool": ["refund_order"]},
            "where": {"ci": True, "runtime": True},
        }
    )
    warnings = validate_rules(load_rules(rules_dir) + [extra])
    assert any(
        "refund_order" in w and "duplicate-forbid" in w for w in warnings
    ), warnings


def test_validate_rules_flags_a_duplicated_idempotency_spec(tmp_path):
    rules_dir = _repo_with_rules(tmp_path)
    extra = parse_rule(
        {
            "rule": "duplicate-idem",
            "then": {"idempotent": {"tool": "refund_order", "key": "${order_id}"}},
            "where": {"ci": True, "runtime": True},
        }
    )
    warnings = validate_rules(load_rules(rules_dir) + [extra])
    assert any("duplicate-idem" in w for w in warnings), warnings


def test_distinct_assertions_do_not_warn(tmp_path):
    """Only *identical* assertions count.  Two rules on the same tool but
    different constraints are legitimate, and flagging them would be noise."""
    first = parse_rule(
        {"rule": "a", "then": {"forbid_tool": ["refund_order"]}, "where": {"ci": True}}
    )
    second = parse_rule(
        {
            "rule": "b",
            "then": {"idempotent": {"tool": "refund_order", "key": "${order_id}"}},
            "where": {"ci": True},
        }
    )
    assert not [
        w for w in validate_rules([first, second]) if "都在声明同一个" in w
    ]


def test_one_owner_of_an_assertion_does_not_warn():
    single = parse_rule(
        {"rule": "only", "then": {"forbid_tool": ["refund_order"]}, "where": {"ci": True}}
    )
    assert not [
        w for w in validate_rules([single]) if "都在声明同一个" in w
    ]


def test_duplicate_assertions_actually_do_double_the_count(tmp_path):
    """The premise of the warning, verified rather than assumed."""
    from agentguard.engine import evaluate_all
    from agentguard.trace import load_trace

    repo = Path(__file__).resolve().parents[1]
    trace = load_trace(repo / "examples/refund_incident/trace.json")
    base = [r for r in load_rules(repo / "rules") if r.is_ci]
    before = sum(len(r.violations) for r in evaluate_all(base, trace))

    duplicate = parse_rule(
        {
            "rule": "duplicate-idem",
            "then": {"idempotent": {"tool": "refund_order", "key": "${order_id}"}},
            "where": {"ci": True, "runtime": True},
            "severity": "critical",
        }
    )
    after = sum(len(r.violations) for r in evaluate_all(base + [duplicate], trace))
    assert after > before, "the duplicate rule did not add a violation"


# --------------------------------------------------------------------------- #
# BUG 42: the promptfoo export carried a variable that nothing consumed.
#
# `when.intent_any` was written to `vars.__agentguard_intent_any`, but no
# assertion read it -- promptfoo has no conditional-assertion primitive.  So the
# config contained something that looked exactly like the `when` gate surviving
# the export, while the gate had in fact been dropped (the exported assertion is
# unconditional).  The orphaned variable contradicted the warning printed
# immediately above it, and a user who trusted the YAML over stderr would ship a
# config that fails on legitimate refund requests.
# --------------------------------------------------------------------------- #
def test_promptfoo_export_has_no_orphaned_when_variable():
    repo = Path(__file__).resolve().parents[1]
    result = export_promptfoo(load_rules(repo / "rules"))

    for test in result.config["tests"]:
        for name in test.get("vars", {}):
            assert not name.startswith("__agentguard"), (
                f"{name!r} is set in vars but no assertion reads it -- it makes "
                "the `when` gate look ported when it was dropped"
            )


def test_promptfoo_export_records_the_unported_gate_as_metadata():
    """The condition must still be discoverable -- just not dressed up as a
    live binding."""
    repo = Path(__file__).resolve().parents[1]
    result = export_promptfoo(load_rules(repo / "rules"))
    gated = [
        t
        for t in result.config["tests"]
        if t.get("metadata", {}).get("agentguard_when")
    ]
    assert gated, "the gated rule lost all trace of its `when` block"
    when = gated[0]["metadata"]["agentguard_when"]
    assert when["ported"] is False
    assert when["intent_any"]


def test_promptfoo_export_warns_that_the_gate_was_widened():
    repo = Path(__file__).resolve().parents[1]
    result = export_promptfoo(load_rules(repo / "rules"))
    assert any("when=" in item for item in result.unsupported), result.unsupported


def test_promptfoo_export_is_valid_yaml(tmp_path):
    import yaml

    from agentguard.exporters.promptfoo import render_promptfoo

    repo = Path(__file__).resolve().parents[1]
    yaml.safe_load(render_promptfoo(load_rules(repo / "rules")))


# --------------------------------------------------------------------------- #
# BUG 43: `max_tool_calls` on a runtime rule produced no warning at all.
#
# `validate_rules` warned about `turn_range`, `memory_persistence` and
# `require_tool` on runtime rules -- all unanswerable from a single call -- but
# forgot `max_tool_calls`, which is equally unanswerable.  A call-budget rule
# with `where.runtime: true` therefore loaded and validated silently and never
# fired: the user believed there was an online call budget when there was none.
# Same failure class the sibling warnings exist to catch.
# --------------------------------------------------------------------------- #
def test_max_tool_calls_on_a_runtime_rule_is_warned_about():
    rule = parse_rule(
        {
            "rule": "budget",
            "then": {"max_tool_calls": 2},
            "where": {"ci": True, "runtime": True},
        }
    )
    warnings = validate_rules([rule])
    assert any("max_tool_calls" in w for w in warnings), warnings


def test_max_tool_calls_on_a_ci_only_rule_is_not_warned_about():
    """The assertion is perfectly valid in CI -- warning there would be noise."""
    rule = parse_rule(
        {
            "rule": "budget",
            "then": {"max_tool_calls": 2},
            "where": {"ci": True, "runtime": False},
        }
    )
    assert not [w for w in validate_rules([rule]) if "max_tool_calls" in w]


def test_a_runtime_rule_that_can_never_fire_is_flagged():
    """`guarded_tools()` comes back empty, which the CLI reads as "no rule
    mentions this tool".  The user must be told the rule is inert, or an
    unguarded ALLOW looks identical to a checked one."""
    rule = parse_rule(
        {
            "rule": "inert",
            "then": {"require_tool": ["verify_identity"]},
            "where": {"ci": True, "runtime": True},
        }
    )
    warnings = validate_rules([rule])
    assert any("等于不存在" in w for w in warnings), warnings


def test_a_runtime_rule_with_a_real_assertion_is_not_flagged_as_inert():
    rule = parse_rule(
        {
            "rule": "real",
            "then": {"forbid_tool": ["refund_order"]},
            "where": {"ci": True, "runtime": True},
        }
    )
    assert not [w for w in validate_rules([rule]) if "等于不存在" in w]


def test_shipped_rules_produce_no_warnings():
    """If the bundled rule set warns, users learn to ignore warnings."""
    repo = Path(__file__).resolve().parents[1]
    assert validate_rules(load_rules(repo / "rules")) == []


# --------------------------------------------------------------------------- #
# BUG 44: `FileStore` allowed a silent double charge across processes.
#
# Atomic *writes* (temp file + os.replace) do not make the *read-modify-write*
# atomic: two processes both read an empty ledger, both decide their key is
# free, and both charge.  Measured with 8 processes reserving one key: 2
# believed they had won -- no error, no corruption, just a duplicate charge
# nobody would notice until reconciliation.  The README called this a "race";
# it was actually a silent correctness failure, which is the exact class this
# project exists to eliminate.
#
# Fixed with an O_CREAT|O_EXCL lock file around the read-modify-write, plus a
# re-read from disk so no process decides on a stale in-memory copy.
# --------------------------------------------------------------------------- #
def _reserve_in_subprocess(path: str, order: str, index: int) -> None:
    from agentguard.idempotency import FileStore

    try:
        store = FileStore(path)
        store.reserve(f"charge:{order}")
    except Exception:  # noqa: BLE001 - subprocess boundary
        pass


def test_file_store_reserves_exactly_once_across_processes(tmp_path):
    """Real subprocesses, because a thread test cannot expose a file-lock bug."""
    import multiprocessing as mp

    path = str(tmp_path / "ledger.json")
    FileStore(path)  # pre-create so every process opens an existing file

    procs = [
        mp.Process(target=_reserve_in_subprocess, args=(path, "A1001", i))
        for i in range(6)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(30)

    ledger = json.loads(Path(path).read_text(encoding="utf-8"))
    assert ledger.get("charge:A1001") == 1, (
        f"ledger says {ledger.get('charge:A1001')} charges went through for one "
        "key -- across processes the reservation is not exclusive, and the two "
        "processes both charged without any error"
    )


def test_file_store_does_not_leave_a_lock_file_behind(tmp_path):
    path = tmp_path / "ledger.json"
    store = FileStore(path)
    store.reserve("k")
    store.release("k")
    store.record("j")
    assert list(tmp_path.glob("*.lock")) == [], (
        "a stale lock file would wedge every later caller"
    )


def test_file_store_sees_writes_made_by_another_instance(tmp_path):
    """The file is the source of truth: a long-lived store must not decide on a
    cached copy after someone else has written."""
    path = tmp_path / "ledger.json"
    first = FileStore(path)
    assert first.reserve("k")

    second = FileStore(path)
    assert not second.reserve("k"), "second instance re-reserved a taken key"
    assert not first.reserve("k"), "first instance decided on a stale cache"


def test_file_store_still_refuses_a_corrupt_ledger(tmp_path):
    """The inter-process work must not weaken the fail-loud property."""
    from agentguard.idempotency import LedgerError

    path = tmp_path / "ledger.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(LedgerError):
        FileStore(path)


# --------------------------------------------------------------------------- #
# BUG 45: `init` scaffolded an incident the default `run`/`gate` never touches.
#
# The incident goes to `examples/` so the clean baseline does not fail out of the
# box -- a reasonable choice.  The cost is a first-run trap: run `init`, wire
# `gate` (defaults) into CI, and the gate is green forever.  A gate that has
# never been seen to fail cannot be told apart from one that cannot fail, which
# is this project's core failure mode.  A clean pass now says so, once, and only
# when the incident is actually present.
# --------------------------------------------------------------------------- #
def _init_in(tmp_path: Path):
    from agentguard.cli import main

    assert main(["init", "--dir", str(tmp_path)]) == 0
    return tmp_path


def test_init_puts_the_incident_outside_the_default_traces_dir(tmp_path):
    """Pinning the intentional design: a clean baseline must stay clean."""
    root = _init_in(tmp_path)
    assert (root / "traces").is_dir()
    assert (root / "examples" / "refund_incident.json").exists()
    assert list((root / "traces").glob("*.json")) == [
        root / "traces" / "baseline-clean.json"
    ]


def test_default_run_passes_and_is_told_about_the_incident(tmp_path, capsys):
    from agentguard.cli import main

    root = _init_in(tmp_path)

    import os

    cwd = os.getcwd()
    try:
        os.chdir(root)
        assert main(["run"]) == 0
    finally:
        os.chdir(cwd)

    out = capsys.readouterr().out
    assert "PASS" in out
    assert "refund_incident.json" in out, (
        "the clean first run did not mention the scaffolded incident, so the "
        "user has no way to learn the gate has never been seen to fail"
    )


def test_the_scaffolded_incident_actually_fails_the_gate(tmp_path):
    """The premise of the hint: `run --traces examples` must really fail."""
    from agentguard.cli import main

    root = _init_in(tmp_path)

    import os

    cwd = os.getcwd()
    try:
        os.chdir(root)
        assert main(["run", "--traces", "examples"]) == 1
    finally:
        os.chdir(cwd)


def test_no_hint_when_the_run_already_failed(tmp_path, capsys):
    """The hint is for a *clean* pass only -- printing it on a failure would be
    noise right when the user is reading real violations."""
    from agentguard.cli import main

    root = _init_in(tmp_path)

    import os

    cwd = os.getcwd()
    try:
        os.chdir(root)
        main(["run", "--traces", "examples"])
    finally:
        os.chdir(cwd)

    out = capsys.readouterr().out
    assert "从没见过失败的门禁" not in out


def test_no_hint_outside_a_scaffolded_project(tmp_path, capsys):
    """A real project has no examples/refund_incident.json -- stay quiet."""
    from agentguard.cli import main

    repo = Path(__file__).resolve().parents[1]

    import os

    cwd = os.getcwd()
    try:
        os.chdir(repo)
        main(["run", "--traces", "traces"])
    finally:
        os.chdir(cwd)

    out = capsys.readouterr().out
    assert "从没见过失败的门禁" not in out


# --------------------------------------------------------------------------- #
# BUG 46: the SARIF driver was missing `informationUri`, which SARIF 2.1.0
# requires.  The schema is not advisory for the consumers this output targets:
# a spec-strict uploader discards the whole run, so the alerts silently do not
# appear -- the failure looks like "AgentGuard found nothing" rather than
# "AgentGuard's report was rejected".
#
# Note on method: an earlier version of this check also flagged unresolvable
# `ruleId`s.  That was a bug in the *check* (a generator consumed the id set
# before comparing), not in the output -- the ids do match.  Retracted rather
# than "fixed".  The tests below assert the property directly so the claim and
# the code cannot drift apart again.
# --------------------------------------------------------------------------- #
def _sarif_for_the_incident() -> dict:
    from agentguard.engine import evaluate_all
    from agentguard.report import render_sarif
    from agentguard.trace import load_trace

    repo = Path(__file__).resolve().parents[1]
    trace = load_trace(repo / "examples/refund_incident/trace.json")
    rules = load_rules(repo / "rules")
    ci_rules = [r for r in rules if r.is_ci]
    runs = [(trace, evaluate_all(ci_rules, trace))]
    return json.loads(render_sarif(runs, rules))


def test_sarif_driver_declares_an_information_uri():
    driver = _sarif_for_the_incident()["runs"][0]["tool"]["driver"]
    assert driver.get("informationUri"), (
        "SARIF 2.1.0 requires tool.driver.informationUri; a strict uploader "
        "drops the run without it"
    )
    assert driver["informationUri"].startswith("http")


def test_every_sarif_rule_id_resolves_against_the_driver_catalogue():
    """Asserts the property the buggy probe got wrong -- directly, so a
    false positive cannot hide a real regression."""
    run = _sarif_for_the_incident()["runs"][0]
    declared = [entry["id"] for entry in run["tool"]["driver"]["rules"]]
    unresolvable = [r["ruleId"] for r in run["results"] if r["ruleId"] not in declared]
    assert not unresolvable, f"these ruleIds have no driver entry: {unresolvable}"


def test_sarif_results_carry_the_fields_a_consumer_needs():
    run = _sarif_for_the_incident()["runs"][0]
    assert run["results"], "expected violations in the bundled incident"
    for result in run["results"]:
        assert result["message"]["text"]
        loc = result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        assert loc


def test_sarif_rules_have_a_full_description_for_the_help_panel():
    rules = _sarif_for_the_incident()["runs"][0]["tool"]["driver"]["rules"]
    for entry in rules:
        assert entry["fullDescription"]["text"], entry["id"]








