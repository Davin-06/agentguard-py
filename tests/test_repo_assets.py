"""Guards for the files shipped in the repository.

These are not unit tests of logic -- they keep the shipped examples and docs
from silently rotting (a stale path in the README, an example that no longer
parses, a stray YAML file that breaks `--rules examples`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentguard.learn import HintConfig
from agentguard.rules import load_rules
from agentguard.trace import load_traces

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_rules_load():
    rules = load_rules(ROOT / "rules")
    names = {r.name for r in rules}
    assert {"refund-safety", "memory-budget-constraint", "payment-idempotency"} <= names


def test_shipped_rules_have_no_validation_warnings():
    """The example rules must be exemplary -- no misconfigurations."""
    from agentguard.rules import validate_rules

    assert validate_rules(load_rules(ROOT / "rules")) == []


def test_shipped_traces_load():
    traces = load_traces(ROOT / "examples")
    assert traces, "examples/ must contain at least one trace"
    assert all(t.steps for t in traces)


def test_clean_baseline_passes_every_rule():
    """`traces/baseline-clean.json` must stay green.

    Its whole purpose is to separate "the gate is broken" from "a regression
    was found".  If this test ever fails, the rule set is over-broad rather
    than the agent being wrong -- which is the most confusing failure mode a
    gate can have.
    """
    from agentguard.executors.ci import run_gate

    outcome = run_gate(ROOT / "rules", ROOT / "traces", fail_on="critical")
    assert outcome.passed, [
        v.message
        for _t, results in outcome.runs
        for r in results
        for v in r.violations
    ]


def test_default_cli_paths_exist():
    """`agentguard run` with no arguments uses these; they must be present."""
    from agentguard.cli import DEFAULT_RULES, DEFAULT_TRACES

    assert (ROOT / DEFAULT_RULES).is_dir()
    assert (ROOT / DEFAULT_TRACES).is_dir()


def test_examples_dir_has_no_stray_yaml():
    """`examples/` is a plausible --rules target; a non-rule YAML there breaks it.

    This is why hints.example.yaml lives in docs/ instead.
    """
    stray = sorted(p.name for p in (ROOT / "examples").rglob("*.y*ml"))
    assert stray == [], f"examples/ should hold only traces and scripts, found: {stray}"


def test_hints_example_is_valid():
    hints = HintConfig.from_file(ROOT / "docs" / "hints.example.yaml")
    assert hints.is_destructive("核销优惠券")
    assert not hints.is_destructive("预览订单")
    # built-ins must still be merged in
    assert hints.is_destructive("refund_order")
    assert hints.is_destructive("退款")


def test_readme_documented_paths_exist():
    """Every path the README tells the user to use must actually exist."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    # Paths the README instructs the user to run against: must exist AND be named.
    documented = (
        "rules",
        "examples/demo.py",
        "examples/refund_incident/trace.json",
        "examples/refund_zh/trace.json",
        "docs/hints.example.yaml",
    )
    for rel in documented:
        assert (ROOT / rel).exists(), f"README references missing path: {rel}"
        assert rel in readme, f"README no longer mentions {rel}"

    # Shipped but not necessarily quoted in the README.
    for rel in ("pyproject.toml", "LICENSE", ".github/workflows/agentguard.yml"):
        assert (ROOT / rel).exists(), f"missing shipped file: {rel}"


def test_demo_runs_without_error(capsys):
    """The demo is the front door; it must not break."""
    import runpy

    # demo.py ends with `raise SystemExit(main())`, so the exit must be caught.
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(ROOT / "examples" / "demo.py"), run_name="__main__")
    assert exc.value.code == 0

    out = capsys.readouterr().out
    assert "BLOCK" in out
    assert "ALLOW" in out


def test_readme_python_imports_exist():
    """Every name the README imports from `agentguard` must be importable.

    The integration guide is copy-pasted verbatim, so a documented name that
    does not exist is a broken first-run for every new user.  This caught
    `decision_zh` / `violation_zh` being reachable only via the internal
    `agentguard.messages` module.
    """
    import re

    import agentguard

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    missing: list[str] = []
    for block in re.findall(r"from agentguard import ([^\n]+)", readme):
        names = [n.strip().split(" as ")[0].strip() for n in block.split(",")]
        for name in filter(None, names):
            if not hasattr(agentguard, name):
                missing.append(name)
    assert not missing, f"README imports names that do not exist: {sorted(set(missing))}"


def test_readme_documents_every_supported_format():
    """`--format` choices must all appear in the README, or users will not know
    they exist."""
    from agentguard.cli import _build_parser

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    parser = _build_parser()
    for action in parser._subparsers._group_actions:  # type: ignore[attr-defined]
        for choice, sub in action.choices.items():
            for sub_action in sub._actions:
                if sub_action.dest == "format":
                    for fmt in sub_action.choices:
                        assert fmt in readme, f"README never mentions --format {fmt}"
