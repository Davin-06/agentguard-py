"""Core data models for AgentGuard.

The whole point of these models is that *one* rule object is consumed by *two*
executors (CI replay and runtime guard).  Keep them free of any executor
specific logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Severity = Literal["info", "warning", "critical"]

SEVERITY_ORDER: dict[str, int] = {"info": 0, "warning": 1, "critical": 2}


# --------------------------------------------------------------------------- #
# Trace side
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation observed (or about to happen) in a trace."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    turn: int = 0
    result: Any = None

    def arg(self, key: str, default: Any = None) -> Any:
        """Support dotted lookups: ``arg("order.id")``."""
        current: Any = self.args
        for part in key.split("."):
            if not isinstance(current, dict):
                return default
            current = current.get(part)
            if current is None:
                return default
        return current


@dataclass(frozen=True)
class Step:
    """One conversational turn plus whatever happened inside it."""

    turn: int
    user: str = ""
    assistant: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    #: Facts that the agent demonstrably recalled / still honoured this turn.
    facts: tuple[str, ...] = ()


@dataclass(frozen=True)
class Trace:
    """A replayable conversation script."""

    trace_id: str
    steps: tuple[Step, ...]
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def tools(self) -> list[str]:
        return [call.name for step in self.steps for call in step.tool_calls]

    def tool_calls(self) -> list[ToolCall]:
        return [call for step in self.steps for call in step.tool_calls]

    def user_text(self) -> str:
        return "\n".join(step.user for step in self.steps)

    def turn(self, turn: int) -> Step | None:
        for step in self.steps:
            if step.turn == turn:
                return step
        return None


# --------------------------------------------------------------------------- #
# Verdict side
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Violation:
    rule: str
    kind: str
    severity: Severity
    message: str
    turn: int | None = None
    tool: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "rule": self.rule,
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
        }
        if self.turn is not None:
            payload["turn"] = self.turn
        if self.tool is not None:
            payload["tool"] = self.tool
        if self.detail:
            payload["detail"] = self.detail
        return payload


@dataclass(frozen=True)
class RuleResult:
    """Outcome of evaluating one rule against one trace."""

    rule: str
    applied: bool
    violations: tuple[Violation, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.violations

    @property
    def worst_severity(self) -> Severity | None:
        if not self.violations:
            return None
        return max(
            (v.severity for v in self.violations),
            key=lambda s: SEVERITY_ORDER.get(s, 0),
        )


@dataclass(frozen=True)
class GateDecision:
    """Result of a runtime pre-execution check.

    ``reason`` is the canonical English message -- it is what reaches production
    logs and ``ToolBlocked`` tracebacks.  ``code`` is the machine-readable
    companion, so callers can branch on the *kind* of refusal instead of
    string-matching prose, and the CLI can render it in Chinese.
    """

    allowed: bool
    reason: str
    rule: str | None = None
    severity: Severity | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    #: One of ``allowed`` / ``forbidden_tool`` / ``tool_not_allowed`` /
    #: ``duplicate_transaction``.
    code: str = "allowed"

    def __bool__(self) -> bool:  # allows ``if not guard.check(...)``
        return self.allowed
