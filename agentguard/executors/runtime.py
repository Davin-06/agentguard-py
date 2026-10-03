"""Runtime guard -- the pre-execution half of "one rule, two executors".

Everything else in the ecosystem tests *after* the fact.  NeMo Guardrails does
ship tool-call interception (``ToolOutputRails``), but it is a guardrail
product first; its rules are not the same asset you run in CI.  Here the rule
object is literally the same one the CI executor reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from ..engine import is_missing, render_template, rule_applies
from ..idempotency import ExecutionStore, MemoryStore
from ..models import GateDecision, Step, Trace
from ..rules import Rule, load_rules, validate_rules

#: Which finding to report when one call violates several assertions at once.
#: Higher wins.  ``duplicate_transaction`` tops the list because it is the one
#: that means *money already moved twice* -- the most actionable thing an
#: operator can be told, and the easiest to lose if a lesser check wins.
_PRIORITY = {
    "duplicate_transaction": 2,
    "forbidden_tool": 1,
    "tool_not_allowed": 0,
}


class _ReservationLost(Exception):
    """Raised internally when another caller already holds an idempotency key.

    Not part of the public surface: :meth:`RuntimeGuard.execute` turns it into
    the same ``ToolBlocked`` a duplicate would produce, so callers see one
    consistent refusal regardless of whether the other call has finished yet.
    """

    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


class RuntimeGuard:
    """Wraps an agent's tool dispatch and refuses dangerous calls."""

    def __init__(
        self,
        rules: list[Rule],
        store: ExecutionStore | None = None,
        *,
        enforce_severity: tuple[str, ...] = ("critical", "warning"),
        enforce_allow_list: bool = True,
    ) -> None:
        self.rules = [r for r in rules if r.is_runtime]
        self.store: ExecutionStore = store or MemoryStore()
        self.enforce_severity = set(enforce_severity)
        #: ``allow_tool`` is a *closed* list: at runtime it means "refuse every
        #: tool that is not named here".  That is high-impact, so it is
        #: switchable -- but it defaults on, because a rule that CI enforces
        #: and runtime silently ignores is exactly the drift AgentGuard exists
        #: to prevent.
        self.enforce_allow_list = enforce_allow_list
        self.audit: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        #: Ledger keys claimed by the *current* in-flight call, for stores that
        #: predate ``reserve``/``release``.  Those stores cannot tell "reserved
        #: but not yet committed" from "committed", so we remember the claim
        #: ourselves and write it to the ledger via ``record`` only once the
        #: tool has actually succeeded.  See :meth:`_reserve`.
        self._pending: list[str] = []
        #: Whether ``self.store`` can claim a key atomically.  When it cannot,
        #: the guard falls back to checking ``seen`` and remembering the claim
        #: on itself (see :meth:`_reserve`).
        self._store_can_reserve = hasattr(self.store, "reserve")
        #: Whether the store can *hand a claim back*.  A store with
        #: ``release`` but no ``reserve`` -- an easy half-upgrade, and one the
        #: reference ``MemoryStore`` itself went through -- can still roll back
        #: properly, so it must not be saddled with the legacy commit-at-the-end
        #: flow.  Rolling a successful charge out of the ledger because the
        #: commit step was skipped is a real defect; this flag is what keeps
        #: rollback working on stores that already know how to do it.
        self._store_can_release = hasattr(self.store, "release")
        self._validate()
        self._warn_about_legacy_store()

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_rules(
        cls,
        target: str | Path,
        store: ExecutionStore | None = None,
        **kwargs: Any,
    ) -> "RuntimeGuard":
        return cls(load_rules(target), store=store, **kwargs)

    def _validate(self) -> None:
        """Surface rules that cannot behave as the author probably expects."""
        self.warnings = validate_rules(
            self.rules, enforce_severity=tuple(sorted(self.enforce_severity))
        )

    def _warn_about_legacy_store(self) -> None:
        """Tell the user when their store cannot do atomic reservations.

        Only worth saying if it changes an outcome: with no idempotency rule
        loaded, the store is never consulted and the warning would be noise.
        """
        if self._store_can_reserve:
            return
        polices_duplicates = any(r.then.idempotent for r in self.rules)
        if not polices_duplicates:
            return
        self.warnings.append(
            f"存储后端 {type(self.store).__name__} 没有实现 reserve()/release()，"
            "已降级为「先查后记」流程：单线程下判定结果正确，但并发下同一笔"
            "交易可能被多次放行。该后端只实现了 seen()/record()，"
            "升级到 MemoryStore/FileStore 或自行补上这两个方法即可恢复并发保护。"
        )

    # -- core -------------------------------------------------------------- #
    def check(
        self,
        tool: str,
        args: dict[str, Any] | None = None,
        *,
        user: str = "",
        turn: int = 0,
        history: Sequence[str] = (),
    ) -> GateDecision:
        """Decide whether a tool call may proceed.  Does not mutate the ledger.

        ``history`` is the user's earlier utterances in this session, oldest
        first.  It matters more than it looks: ``when.intent_any`` is a
        substring match over *the whole session's* user text in CI
        (``Trace.user_text()``).  If the runtime guard only saw the current
        turn, a rule that CI enforces would be trivially bypassable at
        runtime -- just stop mentioning the trigger word in the same turn you
        call the dangerous tool.  Passing ``history`` makes both executors
        evaluate ``when`` over the same haystack.
        """
        args = args or {}
        context = self._context(user, turn, history)

        # Evaluate *every* matching assertion before deciding, instead of
        # returning on the first hit.
        #
        # The first-hit version drifted from CI: on the bundled
        # `refund_incident` trace turn 5, CI reported `duplicate_transaction`
        # while the guard reported only `forbidden_tool`, because `forbid_tool`
        # returned before the ledger was ever consulted.  That is exactly the
        # CI/runtime divergence this project exists to eliminate -- and it hid
        # the *worse* finding: "this is a duplicate charge" matters more than
        # "this tool is not allowed".
        #
        # The refusal is the same either way; what changes is that the reason
        # now matches what CI would have said.  ``code`` reports the most
        # specific finding, with the rest kept in ``detail``.
        findings: list[tuple[int, str, str, dict[str, Any]]] = []

        for rule in self.rules:
            if not rule_applies(rule, context):
                continue
            if rule.severity not in self.enforce_severity:
                continue

            if tool in rule.then.forbid_tool:
                findings.append(
                    (
                        _PRIORITY["forbidden_tool"],
                        "forbidden_tool",
                        f"tool {tool!r} is forbidden by rule {rule.name!r}",
                        {"tool": tool},
                    )
                )

            if (
                self.enforce_allow_list
                and rule.then.allow_tool
                and tool not in rule.then.allow_tool
            ):
                findings.append(
                    (
                        _PRIORITY["tool_not_allowed"],
                        "tool_not_allowed",
                        (
                            f"tool {tool!r} is outside the allow-list "
                            f"{sorted(rule.then.allow_tool)} of rule {rule.name!r}"
                        ),
                        {"allow_tool": sorted(rule.then.allow_tool), "tool": tool},
                    )
                )

            for spec in rule.then.idempotent:
                if tool != spec.tool:
                    continue
                key = render_template(spec.key, args)
                if is_missing(key):
                    continue  # cannot police what we cannot identify
                ledger_key = f"{rule.name}:{spec.tool}:{key!r}"
                if self.store.seen(ledger_key):
                    label = spec.key.strip("${}") or spec.key
                    findings.append(
                        (
                            _PRIORITY["duplicate_transaction"],
                            "duplicate_transaction",
                            (
                                f"duplicate transaction: {spec.tool!r} already executed "
                                f"with {label}={key!r}"
                            ),
                            {
                                "key": key,
                                "key_name": label,
                                "ledger_key": ledger_key,
                            },
                        )
                    )

        if not findings:
            return GateDecision(allowed=True, reason="no rule objected", code="allowed")

        # Highest priority first; ties keep rule declaration order.
        findings.sort(key=lambda item: -item[0])
        _priority, code, reason, detail = findings[0]
        others = sorted({item[1] for item in findings} - {code})
        if others:
            detail = {**detail, "also_violated": others}
        return self._deny_for(context, code, reason, detail)

    def record(self, tool: str, args: dict[str, Any] | None = None) -> None:
        """Call after a tool actually executed, to feed the idempotency ledger."""
        args = args or {}
        for rule in self.rules:
            for spec in rule.then.idempotent:
                if tool != spec.tool:
                    continue
                key = render_template(spec.key, args)
                if is_missing(key):
                    continue
                self.store.record(f"{rule.name}:{spec.tool}:{key!r}")

    def _ledger_key(self, rule: Rule, spec: Any, args: dict[str, Any]) -> tuple[str, str] | None:
        """The ledger key for this (rule, idempotency spec, args), or None when
        the key cannot be resolved from the arguments."""
        key = render_template(spec.key, args)
        if is_missing(key):
            return None  # cannot police what we cannot identify
        return f"{rule.name}:{spec.tool}:{key!r}", key

    def _reserve(self, tool: str, args: dict[str, Any] | None) -> None:
        """Claim every idempotency key this call would touch.

        This is the step that makes the guard correct under concurrency.  It
        used to be missing entirely: ``check`` read the ledger, ``fn`` ran (in
        production, a payment call taking hundreds of milliseconds), and only
        then did ``record`` write to it.  Every caller that arrived inside that
        window saw an empty ledger, so twelve simultaneous charges on one order
        all went through -- a guard that reported itself as idempotent while
        providing no protection at all.  Claiming up front closes the window to
        the width of a single dict operation.

        On failure nothing is left behind: the keys *this* call already claimed
        are handed back before the error propagates, so a lost race cannot leak
        a partial claim and permanently block a legitimate retry.

        Three store generations are handled, because a custom store does not
        upgrade itself in lockstep with this guard:

        * ``reserve`` + ``release`` -- fully atomic; the claim is written and
          rolled back in the store itself.
        * ``release`` only -- cannot claim atomically, but can roll back, so the
          guard writes the claim through ``record`` and rolls it out with
          ``release``.
        * ``seen`` + ``record`` only -- the original documented interface.  The
          claim lives on the guard and reaches the ledger only on success.
          Correct for one thread, racy for several, so
          :meth:`_warn_about_legacy_store` says so out loud rather than
          crashing: breaking a working integration on upgrade is worse than
          degrading.
        """
        args = args or {}
        # Any earlier failure may have left claims behind; start clean so a
        # rollback can never reach into another call's reservations.
        self._pending.clear()
        for rule in self.rules:
            for spec in rule.then.idempotent:
                if tool != spec.tool:
                    continue
                resolved = self._ledger_key(rule, spec, args)
                if resolved is None:
                    continue
                ledger_key, _key = resolved

                if not self._store_can_reserve:
                    if self.store.seen(ledger_key):
                        self._release_keys(self._pending)
                        raise _ReservationLost(ledger_key)
                    if self._store_can_release:
                        # Half-upgraded store: it cannot claim atomically, but
                        # it can hand the claim back.  Put it in the ledger now
                        # so a failed call can roll it out -- that is the only
                        # way `release` can undo anything.
                        self.store.record(ledger_key)  # type: ignore[attr-defined]
                    self._pending.append(ledger_key)
                    continue

                if not self.store.reserve(ledger_key):  # type: ignore[attr-defined]
                    # Lost the race (or a genuine duplicate).  Undo what this
                    # call already claimed so we never leak a partial claim.
                    self._release_keys(self._pending)
                    self._pending.clear()
                    raise _ReservationLost(ledger_key)
                self._pending.append(ledger_key)

    def _release_keys(self, keys: Sequence[str]) -> None:
        """Hand back ledger keys, ignoring stores that cannot release."""
        release = getattr(self.store, "release", None)
        if release is None:
            return
        for key in keys:
            release(key)

    def _commit(self) -> None:
        """Turn this call's claims into permanent ledger entries.

        Atomic stores already wrote the entry at ``reserve`` time, and a
        half-upgraded store that can ``release`` already wrote it too -- for
        those there is nothing left to do.  A store with no ``release`` only
        holds the claim in ``self._pending``, so this is where it finally
        reaches the ledger.
        """
        if self._store_can_reserve or self._store_can_release:
            self._pending.clear()
            return
        for key in self._pending:
            self.store.record(key)
        self._pending.clear()

    def _rollback(self) -> None:
        """Undo this call's claims -- the tool did not complete.

        For a store that can ``release`` the claim was already in the ledger,
        so it must be removed or the order could never be charged.  For a
        minimal ``seen``/``record`` store the claim never left the guard, so
        dropping it *is* the rollback.

        A rollback must never mask the original error: if the store itself
        fails while being cleaned up, the tool's exception is the one the
        caller needs to see.
        """
        if self._store_can_reserve or self._store_can_release:
            try:
                self._release_keys(tuple(self._pending))
            except Exception:  # pragma: no cover - defensive
                pass
        self._pending.clear()

    def execute(
        self,
        tool: str,
        args: dict[str, Any] | None,
        fn: Callable[[], Any],
        *,
        user: str = "",
        turn: int = 0,
        history: Sequence[str] = (),
    ) -> Any:
        """Check -> reserve -> run -> record.  The ergonomic entry point.

        Raises :class:`ToolBlocked` instead of returning, so an existing
        ``return tool(**args)`` dispatch becomes ``guard.execute(name, args, lambda: tool(**args))``.

        The reservation happens *before* ``fn`` runs and is rolled back if
        ``fn`` raises, so a failed charge can be retried while a successful one
        cannot be repeated -- including when many callers arrive at once.
        """
        decision = self.check(tool, args, user=user, turn=turn, history=history)
        self._note(tool, args, decision)
        if not decision.allowed:
            raise ToolBlocked(decision)

        try:
            self._reserve(tool, args)
        except _ReservationLost as lost:
            blocked = self._deny_for(
                self._context(user, turn, history),
                "duplicate_transaction",
                f"duplicate transaction: {tool!r} is already in flight or executed",
                {"ledger_key": lost.key},
            )
            self._note(tool, args, blocked)
            raise ToolBlocked(blocked) from None

        try:
            result = fn()
        except BaseException:
            # The tool never completed, so this was not a transaction.  Free
            # the claim or a retry would be refused as a duplicate -- and the
            # caller would have no way to ever charge this order.
            self._rollback()
            raise
        self._commit()
        return result

    # -- async ------------------------------------------------------------- #
    async def acheck(
        self,
        tool: str,
        args: dict[str, Any] | None = None,
        *,
        user: str = "",
        turn: int = 0,
        history: Sequence[str] = (),
    ) -> GateDecision:
        """Async twin of :meth:`check`.

        The check itself is pure CPU work, so this is a thin wrapper -- it
        exists so an async caller can ``await`` every guard call uniformly
        instead of mixing ``guard.check(...)`` into ``async def`` bodies.
        """
        return self.check(tool, args, user=user, turn=turn, history=history)

    async def aexecute(
        self,
        tool: str,
        args: dict[str, Any] | None,
        fn: Callable[[], Awaitable[Any]],
        *,
        user: str = "",
        turn: int = 0,
        history: Sequence[str] = (),
    ) -> Any:
        """Async twin of :meth:`execute`; ``fn`` must return an awaitable.

        Without this, async agents (LangGraph, the Agents SDK, ...) would have
        to reimplement check/record around every awaited tool -- and the copy
        that forgets ``record`` silently disables the idempotency check.

        The reservation window matters *more* here: ``await fn()`` yields the
        event loop, so a concurrent request can be scheduled during the tool
        call itself rather than only between Python bytecodes.
        """
        decision = self.check(tool, args, user=user, turn=turn, history=history)
        self._note(tool, args, decision)
        if not decision.allowed:
            raise ToolBlocked(decision)

        try:
            self._reserve(tool, args)
        except _ReservationLost as lost:
            blocked = self._deny_for(
                self._context(user, turn, history),
                "duplicate_transaction",
                f"duplicate transaction: {tool!r} is already in flight or executed",
                {"ledger_key": lost.key},
            )
            self._note(tool, args, blocked)
            raise ToolBlocked(blocked) from None

        try:
            result = await fn()
        except BaseException:
            self._rollback()
            raise
        self._commit()
        return result

    def _note(self, tool: str, args: dict[str, Any] | None, decision: GateDecision) -> None:
        self.audit.append(
            {
                "tool": tool,
                "args": args or {},
                "allowed": decision.allowed,
                "reason": decision.reason,
                "code": decision.code,
                "rule": decision.rule,
            }
        )

    # -- helpers ----------------------------------------------------------- #
    def _deny(
        self,
        rule: Rule,
        message: str,
        code: str,
        detail: dict[str, Any] | None = None,
    ) -> GateDecision:
        return GateDecision(
            allowed=False,
            reason=message,
            rule=rule.name,
            severity=rule.severity,
            detail=detail or {},
            code=code,
        )

    def _deny_for(
        self,
        context: Trace,
        code: str,
        message: str,
        detail: dict[str, Any] | None = None,
    ) -> GateDecision:
        """Build a refusal for ``code``, naming the rule that produced it.

        ``check`` collects findings first and decides afterwards, so it no
        longer has the rule object in hand.  Re-deriving it here keeps
        ``GateDecision.rule`` (and therefore the audit trail and the CLI's
        "rule=... severity=..." line) accurate.
        """
        for rule in self.rules:
            if not rule_applies(rule, context):
                continue
            if rule.severity not in self.enforce_severity:
                continue
            if code == "forbidden_tool" and rule.then.forbid_tool:
                return self._deny(rule, message, code, detail)
            if code == "tool_not_allowed" and rule.then.allow_tool:
                return self._deny(rule, message, code, detail)
            if code == "duplicate_transaction" and rule.then.idempotent:
                return self._deny(rule, message, code, detail)

        # Should not happen, but never lose the refusal itself over a missing
        # label -- blocking without a rule name beats allowing silently.
        return GateDecision(
            allowed=False, reason=message, detail=detail or {}, code=code
        )

    @staticmethod
    def _context(user: str, turn: int, history: Sequence[str] = ()) -> Trace:
        """Wrap the runtime context so the same ``when`` logic can be reused.

        Earlier utterances become real steps so that ``when`` sees the same
        session-level haystack CI sees.  They are numbered ``1..N``; the current
        turn keeps its caller-supplied number, or continues the sequence when
        none was given.  Passing both is allowed but ``turn`` should then be
        greater than ``len(history)``.
        """
        prior = tuple(
            Step(turn=index, user=text) for index, text in enumerate(history, 1)
        )
        current_turn = turn if turn > 0 else len(prior) + 1
        return Trace(
            trace_id="runtime",
            steps=prior + (Step(turn=current_turn, user=user),),
            source="runtime",
        )

    def guarded_tools(self) -> dict[str, list[str]]:
        """Introspection: which tools are policed, and by which rules."""
        mapping: dict[str, list[str]] = {}
        for rule in self.rules:
            for tool in sorted(rule.runtime_tools()):
                mapping.setdefault(tool, []).append(rule.name)
        return mapping

    def is_policed(self, tool: str) -> bool:
        """Whether *any* loaded runtime rule mentions this tool.

        A guard that returns ``allowed`` for a tool no rule ever looks at is
        not saying "this is safe" -- it is saying "I have nothing to say".
        Those are different claims, and at the CLI they looked identical:
        checking a misspelled or not-yet-modelled tool printed a clean
        ``[ALLOW]``.  Callers that care about the distinction (the CLI does)
        can ask.
        """
        return bool(self.guarded_tools().get(tool))


class ToolBlocked(RuntimeError):
    """Raised by :meth:`RuntimeGuard.execute` when a call is refused."""

    def __init__(self, decision: GateDecision) -> None:
        super().__init__(decision.reason)
        self.decision = decision
