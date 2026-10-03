#!/usr/bin/env python
"""End-to-end demo: one rule, two executors.

Run it with no arguments:

    python examples/demo.py

It walks the exact incident that motivated AgentGuard:

    1. a production trace where the agent refunded an order nobody asked to refund
    2. `learn` derives candidate rules from that trace (advisory, with confidence)
    3. the same rule blocks the regression in CI
    4. the same rule blocks the call at runtime, before it executes
    5. the same rule exports to promptfoo / NeMo Guardrails

The point being demonstrated: steps 3, 4 and 5 all read *one* YAML file.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentguard.engine import evaluate  # noqa: E402
from agentguard.executors.ci import run_gate  # noqa: E402
from agentguard.executors.runtime import RuntimeGuard, ToolBlocked  # noqa: E402
from agentguard.exporters.nemo import export_nemo  # noqa: E402
from agentguard.exporters.promptfoo import export_promptfoo  # noqa: E402
from agentguard.learn import learn_from_trace  # noqa: E402
from agentguard.rules import load_rules  # noqa: E402
from agentguard.trace import load_trace  # noqa: E402

RULES = ROOT / "rules"
TRACES = ROOT / "examples"


def banner(n: int, title: str) -> None:
    print()
    print("=" * 68)
    print(f"  {n}. {title}")
    print("=" * 68)


def main() -> int:
    print("\nAgentGuard 端到端演示 — 一份规则，两个执行出口")

    # ---------------------------------------------------------------- #
    banner(1, "生产事故：用户只是想查物流，Agent 却退了两次款")
    trace = load_trace(TRACES / "refund_incident" / "trace.json")
    for step in trace.steps:
        calls = ", ".join(c.name for c in step.tool_calls) or "-"
        print(f"  turn {step.turn}: {step.user[:26]:<28} → {calls}")

    # ---------------------------------------------------------------- #
    banner(2, "learn：从 trace 推导候选规则（人工确认，不自动生效）")
    candidates = learn_from_trace(trace)
    for c in candidates:
        print("  " + c.summary())
        for ev in c.evidence[:2]:
            print(f"        · {ev}")
    print("\n  → 注意：每条候选都标记 needs_review=True。")
    print("    判断「哪一步是错的」需要人来确认，AgentGuard 不做一键魔法。")

    # ---------------------------------------------------------------- #
    banner(3, "CI 门禁：同一批规则卡住这次回归")
    outcome = run_gate(RULES, TRACES, fail_on="critical")
    for trace, results in outcome.runs:
        hit = [r for r in results if r.violations]
        for r in hit:
            # Print the trace id: two traces can violate the same rule, and
            # `[FAIL] refund-safety` twice in a row tells the reader nothing.
            print(f"  [FAIL] {trace.trace_id:<26} {r.rule}: {len(r.violations)} 处违规")
    print(f"\n  exit_code = {outcome.exit_code}   （CI 里这就是 merge 被拦住的信号）")

    # ---------------------------------------------------------------- #
    banner(4, "运行时护栏：同一个 YAML，拦在工具执行之前")
    guard = RuntimeGuard.from_rules(RULES)
    for warning in guard.warnings:
        print(f"  warn: {warning}")
    print(f"  被守住的工具: {guard.guarded_tools()}")

    def refund_order(order_id: str) -> str:
        raise AssertionError("这行不应该被执行到！")

    try:
        guard.execute(
            "refund_order",
            {"order_id": "A1001", "amount": 299},
            lambda: refund_order("A1001"),
            user="我上周那单还没到，帮我看下物流",
        )
    except ToolBlocked as exc:
        print(f"\n  [BLOCK] {exc.decision.reason}")
        print(f"          rule={exc.decision.rule}  severity={exc.decision.severity}")
        print("          → 退款函数从未被调用，事故没有发生。")

    safe = guard.check("query_order", {"order_id": "A1001"}, user="帮我看下物流")
    print(f"\n  [{'ALLOW' if safe.allowed else 'BLOCK'}] query_order —— 正常查询不受影响")

    # ---------------------------------------------------------------- #
    banner(5, "导出：不跟生态打架，当翻译层")
    pf = export_promptfoo(load_rules(RULES))
    pf_asserts = [a["type"] for t in pf.config["tests"] for a in t["assert"]]
    print(f"  promptfoo: 生成 {len(pf.config['tests'])} 个 test case")
    print(f"            断言类型 {sorted(set(pf_asserts))}")
    print("            （not-trajectory:tool-used 即「禁止调用该工具」）")
    if pf.unsupported:
        print(f"\n            诚实标注：{len(pf.unsupported)} 条约束 promptfoo 表达不了")
        for item in pf.unsupported[:3]:
            print(f"              - {item}")

    nemo = export_nemo(load_rules(RULES))
    flows = nemo.config["rails"].get("tool_output", {}).get("flows", [])
    print(f"\n  NeMo Guardrails: tool_output.flows = {flows}")
    print("            （必须用内置流程名；写错会让 NeMo 静默禁用全部工具护栏）")
    if nemo.unsupported:
        print(f"            诚实标注：{len(nemo.unsupported)} 条约束 NeMo 表达不了")

    # ---------------------------------------------------------------- #
    banner(6, "结论")
    print("  三个出口 —— CI / runtime / 导出 —— 读的是同一份 rules/*.yaml。")
    print("  这就是 AgentGuard 的全部主张：规则是唯一资产，执行位置只是 where 字段。")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
