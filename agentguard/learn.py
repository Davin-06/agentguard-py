"""``agentguard learn`` -- derive *behavioural rules* from a production trace.

This is the gap that Azure/agentops leaves open on purpose.  Their
``trace_promotion.py`` maps trace fields onto ``{input, expected}`` pairs and
stops; their own spec lists "automatic labeling or correctness judgment" as
*out of scope*.  So the interesting question -- *which step was the wrong one?*
-- is unanswered there.

Answering it requires judgement, and judgement can be wrong.  Therefore this
module is explicitly **advisory**: it emits *candidates* with a confidence
score and a rationale, and nothing is written to a rule file unless the caller
passes ``--apply``.  No "one-click magic" claims.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import ToolCall, Trace
from .rules import IdempotencyRule, Rule, Then, When, Where

#: Tool names that mutate state.  Matched as substrings, case-insensitive.
#: English is the default because most agent tool names are English, but this
#: is only a starting point -- see :class:`HintConfig`.
DESTRUCTIVE_HINTS: tuple[str, ...] = (
    "refund",
    "charge",
    "pay",
    "payment",
    "transfer",
    "delete",
    "remove",
    "drop",
    "cancel",
    "create",
    "purchase",
    "subscribe",
    "unsubscribe",
    "send",
    "execute",
    "commit",
    "deploy",
    "write",
    "update",
    "revoke",
    "grant",
)

#: Chinese tool names, e.g. ``退款`` / ``取消订单``.  Matched as substrings, so
#: ``退款工具`` also matches.
DESTRUCTIVE_HINTS_ZH: tuple[str, ...] = (
    "退款",
    "退单",
    "退货",
    "扣款",
    "付款",
    "支付",
    "转账",
    "打款",
    "删除",
    "移除",
    "清空",
    "取消",
    "下单",
    "创建",
    "新建",
    "购买",
    "订阅",
    "发送",
    "推送",
    "提交",
    "执行",
    "部署",
    "发布",
    "修改",
    "更新",
    "撤销",
    "授权",
    "重置",
    "关闭",
)

#: Read-only tool names -- never proposed as forbidden.
READONLY_HINTS: tuple[str, ...] = (
    "query",
    "get",
    "list",
    "search",
    "find",
    "read",
    "fetch",
    "lookup",
    "status",
    "track",
    "check",
    "describe",
    "show",
)

READONLY_HINTS_ZH: tuple[str, ...] = (
    "查询",
    "获取",
    "查看",
    "搜索",
    "列表",
    "读取",
    "检索",
    "详情",
    "状态",
    "统计",
    "物流",
    "信息",
)

#: Argument names that plausibly identify a transaction.
KEY_HINTS: tuple[str, ...] = (
    "order_id",
    "orderId",
    "transaction_id",
    "txn_id",
    "tx_id",
    "payment_id",
    "refund_id",
    "request_id",
    "idempotency_key",
    "id",
    "key",
    "uuid",
)

#: Chinese transaction identifiers, e.g. ``订单号``.
KEY_HINTS_ZH: tuple[str, ...] = (
    "订单号",
    "订单编号",
    "单号",
    "交易号",
    "交易流水",
    "流水号",
    "支付号",
    "退款单号",
    "编号",
    "序号",
    "卡号",
    "账号",
    "商户号",
    "用户号",
)

ALL_KEY_HINTS: tuple[str, ...] = KEY_HINTS + KEY_HINTS_ZH

_INTENT_STOPWORDS = set(
    "的 了 吗 呢 我 你 他 她 它 是 在 有 和 与 就 都 也 还 不 没 the a an is are was "
    "to of for and or in on at it i you we they my your".split()
)

#: Single characters that never carry intent by themselves.
#:
#: Chinese has no word boundaries, so ``[\u4e00-\u9fff]{2,}`` captures the
#: *whole sentence* -- and a whole sentence is a useless ``intent_any`` value,
#: because ``intent_any`` is a substring match that will never recur.  We cut
#: CJK runs at these characters and keep the surviving fragments.
#:
#: Note what is deliberately *absent*: 到, 货, 单, 退, 款 ... Characters that
#: begin real domain words must stay, or we would shred 到货 / 退款 / 订单.
_CJK_STOPCHARS = set(
    "的了吗呢啊吧哦嗯"
    "我你他她它们咱"
    "是在有和与就都也还不没"
    "这那个什么怎样"
    "想要请帮给把被让使会能可以对为之其"
    "又再很太更只但而或如若则因所并且"
    "看说问找上下起来去过"
)


@dataclass
class Candidate:
    """A proposed rule plus the reasoning that produced it."""

    rule: Rule
    confidence: float
    rationale: str
    evidence: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return True  # by design: nothing auto-applies

    def summary(self) -> str:
        bar = "#" * int(round(self.confidence * 10))
        return f"[{self.confidence:4.0%}] {bar:<10} {self.rule.name} — {self.rationale}"


# --------------------------------------------------------------------------- #
# Heuristics
# --------------------------------------------------------------------------- #
@dataclass
class HintConfig:
    """Which tool names count as destructive / read-only.

    The built-in lists are a starting point, not a spec.  Projects with
    domain-specific or non-English tool names should extend them -- via
    :meth:`from_file` (a YAML/JSON file) or by constructing this directly.

    ``is_destructive`` gives read-only hints priority: a tool named
    ``query_refund`` is a *reader* of refund state, not a mutator.
    """

    destructive: tuple[str, ...] = DESTRUCTIVE_HINTS + DESTRUCTIVE_HINTS_ZH
    readonly: tuple[str, ...] = READONLY_HINTS + READONLY_HINTS_ZH

    def is_destructive(self, tool: str) -> bool:
        lowered = tool.lower()
        if any(hint in lowered for hint in self.readonly):
            return False
        return any(hint in lowered for hint in self.destructive)

    @classmethod
    def from_file(cls, path: str | Path, *, replace: bool = False) -> "HintConfig":
        """Load hints from a YAML/JSON file.

        By default the file **extends** the built-ins, because a project
        usually only needs to add a handful of domain-specific names.
        Pass ``replace=True`` to use only the file's lists.
        """
        import json

        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"hints file not found: {path}") from exc

        data: Any
        if path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            import yaml

            data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: hints file must be a mapping")

        destructive = tuple(str(v) for v in data.get("destructive", []))
        readonly = tuple(str(v) for v in data.get("readonly", []))
        if replace:
            return cls(destructive=destructive, readonly=readonly)
        return cls(
            destructive=DESTRUCTIVE_HINTS + DESTRUCTIVE_HINTS_ZH + destructive,
            readonly=READONLY_HINTS + READONLY_HINTS_ZH + readonly,
        )


_DEFAULT_HINTS = HintConfig()


def is_destructive(tool: str, hints: HintConfig | None = None) -> bool:
    return (hints or _DEFAULT_HINTS).is_destructive(tool)


def _split_cjk(run: str) -> list[str]:
    """Cut a CJK run into maximal fragments free of stopword characters."""
    fragments: list[str] = []
    current: list[str] = []
    for ch in run:
        if ch in _CJK_STOPCHARS:
            if len(current) >= 2:
                fragments.append("".join(current))
            current = []
        else:
            current.append(ch)
    if len(current) >= 2:
        fragments.append("".join(current))
    return fragments


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _tokens(text: str) -> list[str]:
    """Extract intent keywords that can actually serve as ``intent_any`` values.

    The previous implementation returned whole sentences for Chinese input
    (``我上周那单还没到``), which made every learned rule untriggerable: the
    rule's ``when`` never matched, so the rule never fired.
    """
    found: list[str] = []
    for run in re.findall(r"[A-Za-z_]{3,}|[\u4e00-\u9fff]+", text.lower()):
        if run.isascii():
            if run not in _INTENT_STOPWORDS:
                found.append(run)
            continue
        for fragment in _split_cjk(run):
            if len(fragment) <= 3:
                found.append(fragment)
            else:
                # A long run-on is probably several words fused together
                # (查询订单号) -- take bigrams so the real words survive.
                found.extend(fragment[i : i + 2] for i in range(len(fragment) - 1))
    return _dedupe(found)


def _guess_key(calls: list[ToolCall]) -> str | None:
    """Pick the argument that most likely identifies the transaction."""
    if not calls:
        return None
    shared: set[str] = set(calls[0].args)
    for call in calls[1:]:
        shared &= set(call.args)
    for hint in ALL_KEY_HINTS:
        for name in sorted(shared):
            if name == hint or name.lower() == hint.lower():
                return name
    for hint in ALL_KEY_HINTS:
        for call in calls:
            for name in call.args:
                if hint in name.lower():
                    return name
    return None


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def learn_from_trace(
    trace: Trace,
    *,
    hints: HintConfig | None = None,
    max_calls_before_flag: int = 6,
) -> list[Candidate]:
    hints = hints or _DEFAULT_HINTS
    candidates: list[Candidate] = []
    candidates.extend(_learn_forbidden_tools(trace, hints))
    candidates.extend(_learn_idempotency(trace, hints))
    candidates.extend(_learn_call_budget(trace, max_calls_before_flag))
    return sorted(candidates, key=lambda c: c.confidence, reverse=True)


def _learn_forbidden_tools(trace: Trace, hints: HintConfig) -> list[Candidate]:
    """Destructive tool invoked during an informational conversation."""
    all_calls = trace.tool_calls()
    destructive = [c for c in all_calls if hints.is_destructive(c.name)]
    if not destructive:
        return []

    readonly = [c for c in all_calls if not hints.is_destructive(c.name)]
    readonly_names = sorted({c.name for c in readonly})
    destructive_names = sorted({c.name for c in destructive})

    # `when.intent_any` is supposed to describe the *read-only* situation in
    # which the destructive call was wrong.  A token like `退款` is the
    # opposite: it is how a user *asks* for the destructive action.  Leaving it
    # in produces a rule that fires when the user requests a refund -- and then
    # forbids the refund.  In `refund_incident` the token survives because turn
    # 4 says "我没说要退款啊", a negation that happens to look like a request.
    # Drop destructive vocabulary; keep read-only vocabulary (that is the
    # signal we actually want) and the rest of the user's wording.
    readonly_names = set(readonly_names)
    destructive_vocab = tuple(h.lower() for h in hints.destructive)
    intent_tokens = [
        t
        for t in _tokens(trace.user_text())
        if t not in readonly_names and not any(v in t for v in destructive_vocab)
    ]
    intent_tokens = intent_tokens[:6]

    # Confidence: a destructive call during a purely informational flow is the
    # clearest possible signal.  Reduce it when we cannot describe the intent.
    confidence = 0.82 if readonly_names else 0.55
    if not intent_tokens:
        # An empty `when` means "always applies", so the rule would forbid the
        # tool unconditionally -- including legitimate uses.  Drop below the
        # default 0.6 threshold so `--apply` will not write it silently.
        confidence = min(confidence, 0.5)
    if len(destructive_names) > 2:
        confidence -= 0.1

    evidence = [
        f"turn {c.turn}: {c.name}({c.args})" for c in destructive
    ]
    rationale = (
        f"检测到只读意图的会话中调用了破坏性工具 {destructive_names}"
        if readonly_names
        else f"调用了破坏性工具 {destructive_names}，但无法确认意图"
    )
    if not intent_tokens:
        rationale += "（无法从对话中提取意图关键词，需人工补充 when）"

    # `forbid_tool` and `allow_tool` describe the same fact here, and emitting
    # both made every bad call show up twice in the report -- once as
    # "forbidden", once as "outside the allow-list".
    #
    # Worse, the two are not equally safe.  `allow_tool` is a *closed* list: at
    # runtime it refuses every tool that is not named.  The learner only ever saw
    # the handful of tools this one trace happened to use, so promoting that
    # observation to a closed list would block tools this agent legitimately has
    # (`refund_incident` yields allow_tool: [query_order] -- which would refuse
    # get_logistics).  Forbidding the specific tool that actually misbehaved is
    # the narrow, defensible claim, so that is what we emit.
    rule = Rule(
        name=f"{_slug(trace.trace_id)}-forbid-{_slug(destructive_names[0])}",
        then=Then(forbid_tool=tuple(destructive_names)),
        when=When(intent_any=tuple(intent_tokens)),
        where=Where(ci=True, runtime=True),
        severity="critical",
        description=(
            "由 trace 推导：只读场景下不得调用破坏性工具。"
            "如需收紧为「除名单外全拒」，请人工补充 allow_tool——"
            "learner 只见过本条 trace 用到的工具，自动生成会误伤。"
            if intent_tokens
            else (
                "由 trace 推导，但未能提取到意图关键词。本规则没有 when 条件，"
                "因此会对所有场景生效（含合法的退款/扣款）。"
                "请手动补充 when.intent_any 后再启用。"
            )
        ),
        source_trace=trace.trace_id,
    )
    return [
        Candidate(
            rule=rule,
            confidence=max(0.1, min(0.95, confidence)),
            rationale=rationale,
            evidence=evidence,
        )
    ]


def _learn_idempotency(trace: Trace, hints: HintConfig) -> list[Candidate]:
    """Same destructive tool + same identifying argument more than once.

    Read-only tools are deliberately excluded: calling ``get_logistics`` twice
    is not a duplicate transaction, it is a user asking twice.  Only tools that
    mutate state can meaningfully violate idempotency.
    """
    by_tool: dict[str, list[ToolCall]] = {}
    for call in trace.tool_calls():
        if not hints.is_destructive(call.name):
            continue
        by_tool.setdefault(call.name, []).append(call)

    out: list[Candidate] = []
    for tool, calls in sorted(by_tool.items()):
        if len(calls) < 2:
            continue
        key = _guess_key(calls)
        if not key:
            continue
        seen: dict[str, int] = {}
        duplicate_turns: list[str] = []
        for call in calls:
            value = call.arg(key)
            if value is None:
                continue
            token = repr(value)
            if token in seen:
                duplicate_turns.append(
                    f"turns {seen[token]} & {call.turn} share {key}={value!r}"
                )
            else:
                seen[token] = call.turn
        if not duplicate_turns:
            continue

        confidence = 0.9 if key.lower().endswith("id") else 0.7
        rule = Rule(
            name=f"{_slug(trace.trace_id)}-idempotent-{_slug(tool)}",
            then=Then(idempotent=(IdempotencyRule(tool=tool, key=f"${{{key}}}"),)),
            where=Where(ci=True, runtime=True),
            severity="critical",
            description=(
                f"由 trace 推导：{tool} 对同一 {key} 不得重复执行。"
            ),
            source_trace=trace.trace_id,
        )
        out.append(
            Candidate(
                rule=rule,
                confidence=confidence,
                rationale=f"{tool} 用同一 {key} 被调用了多次",
                evidence=duplicate_turns,
            )
        )
    return out


def _learn_call_budget(trace: Trace, threshold: int) -> list[Candidate]:
    total = len(trace.tool_calls())
    if total <= threshold:
        return []
    rule = Rule(
        name=f"{_slug(trace.trace_id)}-call-budget",
        then=Then(max_tool_calls=threshold),
        where=Where(ci=True, runtime=False),
        severity="warning",
        description=f"由 trace 推导：单次会话工具调用不得超过 {threshold} 次（疑似循环）。",
        source_trace=trace.trace_id,
    )
    # Confidence scales with how far past the threshold the trace went.  A flat
    # value used to sit below the CLI's default --min-confidence of 0.6, which
    # made this candidate permanently unreachable -- a feature that never fired.
    overshoot = total - threshold
    confidence = min(0.8, 0.62 + 0.03 * overshoot)
    return [
        Candidate(
            rule=rule,
            confidence=confidence,
            rationale=f"本次会话共 {total} 次工具调用，超过阈值 {threshold}",
            evidence=[f"total_tool_calls={total}"],
        )
    ]


def _slug(text: str) -> str:
    """Make a rule-name-safe slug, keeping CJK characters.

    Stripping non-ASCII turned a Chinese trace id into the literal string
    "rule", so every learned rule from a Chinese trace was named
    ``rule-forbid-rule`` -- indistinguishable from one another.
    """
    cleaned = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "-", text).strip("-").lower()
    return cleaned or "rule"
