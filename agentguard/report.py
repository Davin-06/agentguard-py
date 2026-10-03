"""Report rendering: terminal, Markdown (for CI artifacts / PR comments),
JSON and SARIF."""

from __future__ import annotations

import json
from typing import Any

from .messages import violation_zh
from .models import RuleResult, SEVERITY_ORDER, Trace, Violation
from .trace import WARNINGS_KEY

_ICON = {"critical": "BLOCK", "warning": "WARN ", "info": "INFO "}

#: SARIF severity levels.  ``critical`` maps to ``error`` because a gate that
#: fails the build must show up red in the GitHub Security tab.
_SARIF_LEVEL = {"critical": "error", "warning": "warning", "info": "note"}

#: (trace, per-trace rule results)
RunResult = list[tuple[Trace, list[RuleResult]]]


def _collect(runs: RunResult) -> tuple[list[Violation], int, int]:
    violations: list[Violation] = []
    applied = 0
    total = 0
    for _trace, results in runs:
        for result in results:
            total += 1
            if result.applied:
                applied += 1
            violations.extend(result.violations)
    return violations, applied, total


def worst(violations: list[Violation]) -> str | None:
    if not violations:
        return None
    return max((v.severity for v in violations), key=lambda s: SEVERITY_ORDER.get(s, 0))


def render_terminal(runs: RunResult) -> str:
    violations, applied, total = _collect(runs)
    lines: list[str] = ["", "AgentGuard — 回归门禁报告", "=" * 64]

    for trace, results in runs:
        n_applied = sum(1 for r in results if r.applied)
        lines.append(
            f"  trace  {trace.trace_id:<26} steps={len(trace.steps):<3} "
            f"tool_calls={len(trace.tool_calls()):<3} rules_hit={n_applied}"
        )
        # A trace we could only partly read must not look like a clean one.
        # stderr alone is not enough: CI logs often keep stdout only.
        for warning in trace.metadata.get(WARNINGS_KEY, []):
            lines.append(f"         [!] {warning}")
    # Label it as rule×trace pairs: with N rules and M traces the denominator
    # is N*M, which "rules applied" misdescribed.
    lines.append(f"  规则×用例命中 {applied}/{total}")
    lines.append("")

    if not violations:
        lines.append("  PASS — 没有规则被违反")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"  FAIL — {len(violations)} 处违规")
    # Group by trace.  Without the trace id a log line like
    # `[BLOCK] refund-safety (第 5 轮)` is ambiguous the moment two traces hit
    # the same rule -- which is the normal case, not an edge case.
    for trace, results in runs:
        trace_violations = [v for r in results for v in r.violations]
        if not trace_violations:
            continue
        lines.append("-" * 64)
        lines.append(f"  trace  {trace.trace_id}  —  {len(trace_violations)} 处")
        for v in trace_violations:
            icon = _ICON.get(v.severity, "?????")
            location = f"第 {v.turn} 轮" if v.turn is not None else "—"
            lines.append(f"  [{icon}] {v.rule}  ({location})")
            lines.append(
                "           "
                + violation_zh(
                    kind=v.kind,
                    message=v.message,
                    detail=v.detail,
                    turn=v.turn,
                    tool=v.tool,
                )
            )
            if v.detail.get("args"):
                lines.append(f"           args={v.detail['args']}")
    lines.append("")
    return "\n".join(lines)


def render_markdown(runs: RunResult) -> str:
    violations, applied, total = _collect(runs)
    status = "✅ PASS" if not violations else "❌ FAIL"
    lines = [
        "## AgentGuard 回归门禁",
        "",
        f"**{status}** — {applied}/{total} 条规则命中，{len(violations)} 处违规",
        "",
        "| Trace | Steps | Tool calls |",
        "| --- | ---: | ---: |",
    ]
    for trace, _results in runs:
        lines.append(
            f"| `{trace.trace_id}` | {len(trace.steps)} | {len(trace.tool_calls())} |"
        )

    if violations:
        lines += [
            "",
            "### 违规明细",
            "",
            "| Trace | 严重度 | 规则 | 位置 | 工具 | 说明 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for trace, results in runs:
            for result in results:
                for v in result.violations:
                    location = f"第 {v.turn} 轮" if v.turn is not None else "—"
                    tool = f"`{v.tool}`" if v.tool else "—"
                    note = violation_zh(
                        kind=v.kind,
                        message=v.message,
                        detail=v.detail,
                        turn=v.turn,
                        tool=v.tool,
                    )
                    lines.append(
                        f"| `{trace.trace_id}` | {v.severity} | `{v.rule}` | "
                        f"{location} | {tool} | {note} |"
                    )
    lines.append("")
    return "\n".join(lines)


def render_sarif(runs: RunResult, rules: list[Any] | None = None) -> str:
    """Render as SARIF 2.1.0, so GitHub surfaces violations in the Security tab.

    Two caveats, stated rather than hidden:

    * **Results are file-level.**  We know the trace file and the turn, but not
      the byte offset of that turn inside the JSON -- ``json.loads`` discards
      position information.  So there is no ``region``, and the turn goes into
      the message instead.  Real line numbers would need a position-aware trace
      parser; a fabricated ``startLine`` would put the marker on a random line.
    * **``message.text`` is the Chinese rendering**, matching the terminal and
      Markdown reports.  The canonical English message is kept in
      ``properties.canonicalMessage`` so log correlation still works.

    ``rules`` is optional: when given, rule descriptions land in the SARIF rule
    metadata, which is what GitHub shows in the alert's help panel.
    """
    from . import __version__

    described = {
        name: (getattr(rule, "description", "") or "").strip()
        for rule in (rules or [])
        if (name := getattr(rule, "name", None))
    }
    severities = {
        name: getattr(rule, "severity", "warning")
        for rule in (rules or [])
        if (name := getattr(rule, "name", None))
    }

    def _rule_entry(name: str, severity: str | None = None) -> dict[str, Any]:
        summary = described.get(name) or f"AgentGuard rule {name}"
        return {
            "id": name,
            "name": name,
            "shortDescription": {"text": summary},
            # GitHub renders `fullDescription` in the alert's help panel.  A
            # rule's `description` in the YAML is written to explain *why* the
            # rule exists (the bundled ones narrate the incident that produced
            # them), so putting it here is what makes the alert actionable
            # rather than just "rule X failed".
            "fullDescription": {"text": summary},
            "defaultConfiguration": {
                "level": _SARIF_LEVEL.get(severity or severities.get(name, ""), "warning")
            },
        }

    # Seed the catalogue from the rules that were evaluated, so a clean run
    # still documents which ruleset was applied -- otherwise `results: []` with
    # `rules: []` looks like nothing ran at all.
    seen_rules: dict[str, dict[str, Any]] = {
        name: _rule_entry(name) for name in described
    }
    results: list[dict[str, Any]] = []

    for trace, rule_results in runs:
        for rule_result in rule_results:
            for v in rule_result.violations:
                seen_rules.setdefault(v.rule, _rule_entry(v.rule, v.severity))

                note = violation_zh(
                    kind=v.kind,
                    message=v.message,
                    detail=v.detail,
                    turn=v.turn,
                    tool=v.tool,
                )
                properties: dict[str, Any] = {
                    "kind": v.kind,
                    "severity": v.severity,
                    "traceId": trace.trace_id,
                    # Canonical English, so a SARIF alert can be matched against
                    # the same message that appears in runtime logs.
                    "canonicalMessage": v.message,
                }
                if v.turn is not None:
                    properties["turn"] = v.turn
                if v.tool is not None:
                    properties["tool"] = v.tool

                results.append(
                    {
                        "ruleId": v.rule,
                        "level": _SARIF_LEVEL.get(v.severity, "warning"),
                        "message": {
                            "text": (
                                f"第 {v.turn} 轮：{note}"
                                if v.turn is not None
                                else note
                            )
                        },
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": trace.source}
                                }
                            }
                        ],
                        "properties": properties,
                    }
                )

    payload = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "AgentGuard",
                        # `informationUri` is required by SARIF 2.1.0 (see
                        # toolComponent.informationUri).  Validators reject a
                        # driver without it, so a spec-strict consumer -- some
                        # CI SARIF uploaders, and GitHub's own schema check --
                        # would discard the whole run rather than show the
                        # alerts.  Emitting it costs nothing.
                        "informationUri": "https://github.com/agentguard/agentguard",
                        "version": __version__,
                        "rules": list(seen_rules.values()),
                    }
                },
                "results": results,
            }
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def to_dict(runs: RunResult) -> dict[str, Any]:
    violations, applied, total = _collect(runs)
    return {
        "version": 1,
        "summary": {
            "passed": not violations,
            "worst_severity": worst(violations),
            "violations": len(violations),
            "rules_applied": applied,
            "rules_total": total,
            "traces": len(runs),
        },
        "runs": [
            {
                "trace_id": trace.trace_id,
                "steps": len(trace.steps),
                "tool_calls": [c.name for c in trace.tool_calls()],
                "rules": [
                    {
                        "rule": r.rule,
                        "applied": r.applied,
                        "passed": r.passed,
                        "violations": [v.to_dict() for v in r.violations],
                    }
                    for r in results
                ],
            }
            for trace, results in runs
        ],
    }
