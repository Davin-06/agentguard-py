"""AgentGuard — 一份规则，两个执行出口。

The public surface is intentionally small::

    from agentguard import load_rules, RuntimeGuard, run_gate

    # CI side
    outcome = run_gate("rules/", "traces/", fail_on="critical")

    # runtime side -- same rule files
    guard = RuntimeGuard.from_rules("rules/")
    guard.execute("refund_order", {"order_id": "A1001"}, do_refund)

Pass ``history`` (earlier user utterances) to ``check`` / ``execute`` whenever a
rule uses ``when.intent_any``: CI matches the whole session, and without the
history the runtime guard only sees the current turn.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .engine import evaluate, evaluate_all, render_template, rule_applies
from .executors.ci import GateOutcome, run_gate, write_reports
from .executors.runtime import RuntimeGuard, ToolBlocked
from .idempotency import ExecutionStore, FileStore, LedgerError, MemoryStore
from .learn import Candidate, learn_from_trace
from .messages import decision_zh, violation_zh
from .models import (
    GateDecision,
    RuleResult,
    Severity,
    Step,
    ToolCall,
    Trace,
    Violation,
)
from .rules import Rule, RuleError, load_rule, load_rules, parse_rule
from .trace import TraceError, load_trace, load_traces, parse_trace

__all__ = [
    "__version__",
    # models
    "Trace",
    "Step",
    "ToolCall",
    "Violation",
    "RuleResult",
    "GateDecision",
    "Severity",
    # rules
    "Rule",
    "RuleError",
    "load_rule",
    "load_rules",
    "parse_rule",
    # traces
    "TraceError",
    "load_trace",
    "load_traces",
    "parse_trace",
    # engine
    "evaluate",
    "evaluate_all",
    "rule_applies",
    "render_template",
    # executors
    "run_gate",
    "write_reports",
    "GateOutcome",
    "RuntimeGuard",
    "ToolBlocked",
    # idempotency
    "ExecutionStore",
    "MemoryStore",
    "FileStore",
    "LedgerError",
    # learning
    "learn_from_trace",
    "Candidate",
    # localisation -- the canonical `reason` / `message` stay English (stable
    # identifiers for logs and alerting); these render the same facts in
    # Chinese for terminal output and Markdown reports.
    "decision_zh",
    "violation_zh",
]
