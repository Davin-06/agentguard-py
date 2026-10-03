"""Rule loading and validation.

A rule is the single first-class asset of AgentGuard: the same YAML file is
consumed by the CI executor *and* by the runtime guard.  ``where`` decides
which executor is allowed to enforce it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import Severity

try:  # PyYAML is the only third-party dependency, and it is optional.
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only without PyYAML
    yaml = None  # type: ignore[assignment]


class RuleError(ValueError):
    """Raised when a rule file is malformed."""


# --------------------------------------------------------------------------- #
# Recognised keys -- unknown ones are rejected rather than ignored.
#
# Silent ignore is a genuine footgun: the original design doc used
# ``when.intent``, which is not a real key.  Under a lenient parser that rule
# loads fine, ``when`` becomes empty, and ``When.always`` turns it into
# "applies unconditionally" -- the opposite of what the author wrote.
# --------------------------------------------------------------------------- #
_TOP_LEVEL_KEYS = {
    "version",
    "rule",
    "name",
    "description",
    "source_trace",
    "when",
    "then",
    "where",
    "severity",
}
_WHEN_KEYS = {"intent_any", "turn_range"}
_THEN_KEYS = {
    "forbid_tool",
    "allow_tool",
    "require_tool",
    "max_tool_calls",
    "idempotent",
    "memory_persistence",
}
_WHERE_KEYS = {"ci", "runtime"}
_IDEMPOTENT_KEYS = {"tool", "key"}
_MEMORY_KEYS = {"fact", "inject_at", "must_recall_at"}


def _reject_unknown(
    data: dict[str, Any], allowed: set[str], *, where: str, rule_name: str
) -> None:
    unknown = sorted(set(data) - allowed)
    if not unknown:
        return
    import difflib

    hints = []
    for key in unknown:
        close = difflib.get_close_matches(key, sorted(allowed), n=1, cutoff=0.6)
        hints.append(f"{key!r}" + (f"（是否想写 {close[0]!r}？）" if close else ""))
    raise RuleError(
        f"规则 {rule_name!r} 的 `{where}` 含无法识别的键：{'、'.join(hints)}。"
        f"可用键：{sorted(allowed)}"
    )


# --------------------------------------------------------------------------- #
# Sub-structures
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class When:
    """Activation condition.  Empty ``When`` means "always applies"."""

    intent_any: tuple[str, ...] = ()
    turn_range: tuple[int, int] | None = None

    @property
    def always(self) -> bool:
        return not self.intent_any and self.turn_range is None


@dataclass(frozen=True)
class IdempotencyRule:
    tool: str
    key: str

    @staticmethod
    def from_raw(raw: Any) -> "IdempotencyRule":
        if not isinstance(raw, dict):
            raise RuleError("`idempotent` must be a mapping with `tool` and `key`")
        tool = raw.get("tool")
        key = raw.get("key")
        if not tool or not key:
            raise RuleError("`idempotent` requires both `tool` and `key`")
        return IdempotencyRule(tool=str(tool), key=str(key))


@dataclass(frozen=True)
class MemoryRule:
    fact: str
    inject_at: int = 1
    must_recall_at: tuple[int, ...] = ()

    @staticmethod
    def from_raw(raw: Any) -> "MemoryRule":
        if not isinstance(raw, dict):
            raise RuleError("`memory_persistence` must be a mapping")
        fact = raw.get("fact")
        if not fact:
            raise RuleError("`memory_persistence` requires `fact`")
        recall = raw.get("must_recall_at") or []
        if not isinstance(recall, (list, tuple)):
            raise RuleError("`must_recall_at` must be a list of turn numbers")
        return MemoryRule(
            fact=str(fact),
            inject_at=int(raw.get("inject_at", 1)),
            must_recall_at=tuple(int(t) for t in recall),
        )


@dataclass(frozen=True)
class Then:
    """The constraints to enforce."""

    forbid_tool: tuple[str, ...] = ()
    allow_tool: tuple[str, ...] = ()
    require_tool: tuple[str, ...] = ()
    max_tool_calls: int | None = None
    idempotent: tuple[IdempotencyRule, ...] = ()
    memory_persistence: tuple[MemoryRule, ...] = ()

    @property
    def empty(self) -> bool:
        return not any(
            (
                self.forbid_tool,
                self.allow_tool,
                self.require_tool,
                self.max_tool_calls is not None,
                self.idempotent,
                self.memory_persistence,
            )
        )


@dataclass(frozen=True)
class Where:
    """Which executors may enforce this rule."""

    ci: bool = True
    runtime: bool = True


@dataclass(frozen=True)
class Rule:
    name: str
    then: Then
    when: When = field(default_factory=When)
    where: Where = field(default_factory=Where)
    severity: Severity = "critical"
    description: str = ""
    source_trace: str | None = None
    version: int = 1
    path: Path | None = None

    # -- convenience ------------------------------------------------------- #
    @property
    def is_ci(self) -> bool:
        return self.where.ci

    @property
    def is_runtime(self) -> bool:
        return self.where.runtime

    def runtime_tools(self) -> set[str]:
        """Tool names this rule can police at runtime (pre-execution)."""
        guarded = set(self.then.forbid_tool)
        guarded |= {item.tool for item in self.then.idempotent}
        return guarded


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _require_yaml() -> None:
    if yaml is None:  # pragma: no cover
        raise RuleError(
            "PyYAML is required to read rule files. Install it with: pip install pyyaml"
        )


def parse_rule(data: dict[str, Any], *, path: Path | None = None) -> Rule:
    if not isinstance(data, dict):
        raise RuleError("rule document must be a mapping")

    name = data.get("rule") or data.get("name")
    if not name:
        raise RuleError("rule document requires a `rule:` name")
    _reject_unknown(data, _TOP_LEVEL_KEYS, where="顶层", rule_name=str(name))

    then_raw = data.get("then")
    if not isinstance(then_raw, dict):
        raise RuleError(f"rule {name!r} requires a `then:` mapping")
    _reject_unknown(then_raw, _THEN_KEYS, where="then", rule_name=str(name))

    def _as_tuple(value: Any, key: str) -> tuple[str, ...]:
        if value in (None, ""):
            return ()
        if isinstance(value, str):
            return (value,)
        if isinstance(value, (list, tuple)):
            return tuple(str(v) for v in value)
        raise RuleError(f"`{key}` must be a string or list of strings")

    max_calls = then_raw.get("max_tool_calls")
    then = Then(
        forbid_tool=_as_tuple(then_raw.get("forbid_tool"), "forbid_tool"),
        allow_tool=_as_tuple(then_raw.get("allow_tool"), "allow_tool"),
        require_tool=_as_tuple(then_raw.get("require_tool"), "require_tool"),
        max_tool_calls=int(max_calls) if max_calls is not None else None,
        idempotent=tuple(
            IdempotencyRule.from_raw(_check_inner(item, _IDEMPOTENT_KEYS, str(name)))
            for item in _as_tuple_raw(then_raw.get("idempotent"), "idempotent")
        ),
        memory_persistence=tuple(
            MemoryRule.from_raw(_check_inner(item, _MEMORY_KEYS, str(name)))
            for item in _as_tuple_raw(
                then_raw.get("memory_persistence"), "memory_persistence"
            )
        ),
    )
    if then.empty:
        raise RuleError(f"rule {name!r} has an empty `then:` block")

    when_raw = data.get("when") or {}
    if not isinstance(when_raw, dict):
        raise RuleError("`when` must be a mapping")
    _reject_unknown(when_raw, _WHEN_KEYS, where="when", rule_name=str(name))
    turn_range_raw = when_raw.get("turn_range")
    turn_range = None
    if turn_range_raw is not None:
        if not isinstance(turn_range_raw, (list, tuple)) or len(turn_range_raw) != 2:
            raise RuleError("`turn_range` must be a two-element list")
        turn_range = (int(turn_range_raw[0]), int(turn_range_raw[1]))
        if turn_range[0] > turn_range[1]:
            raise RuleError(
                f"`turn_range` 起点大于终点：{turn_range}；该规则永远不会生效"
            )
    when = When(
        intent_any=_as_tuple(when_raw.get("intent_any"), "intent_any"),
        turn_range=turn_range,
    )

    where_raw = data.get("where") or {}
    if not isinstance(where_raw, dict):
        raise RuleError("`where` must be a mapping")
    _reject_unknown(where_raw, _WHERE_KEYS, where="where", rule_name=str(name))
    where = Where(
        ci=bool(where_raw.get("ci", True)),
        runtime=bool(where_raw.get("runtime", True)),
    )

    severity = str(data.get("severity", "critical"))
    if severity not in ("info", "warning", "critical"):
        raise RuleError(f"unknown severity {severity!r}")

    return Rule(
        name=str(name),
        then=then,
        when=when,
        where=where,
        severity=severity,  # type: ignore[arg-type]
        description=str(data.get("description", "")),
        source_trace=data.get("source_trace"),
        version=int(data.get("version", 1)),
        path=path,
    )


def _as_tuple_raw(value: Any, key: str) -> tuple[Any, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, dict):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(value)
    raise RuleError(f"`{key}` must be a mapping or list of mappings")


def _check_inner(item: Any, allowed: set[str], rule_name: str) -> Any:
    """Validate the keys of a nested mapping (idempotent / memory entries)."""
    if isinstance(item, dict):
        _reject_unknown(item, allowed, where="then（嵌套项）", rule_name=rule_name)
    return item


def _read_documents(path: Path) -> list[dict[str, Any]]:
    """Read every YAML document in a file.

    A rule file may legitimately hold several rules separated by ``---``
    (``agentguard learn --apply`` writes exactly that), so we must use
    ``safe_load_all``.  ``safe_load`` raises ``ComposerError`` on a
    multi-document stream, which would make learned rule files unloadable.
    """
    _require_yaml()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuleError(f"rule file not found: {path}") from exc
    try:
        documents = [d for d in yaml.safe_load_all(text) if d is not None]
    except yaml.YAMLError as exc:  # type: ignore[union-attr]
        raise RuleError(f"{path}: invalid YAML: {exc}") from exc
    if not documents:
        raise RuleError(f"{path}: empty rule document")
    return documents


def load_rule(path: str | Path) -> Rule:
    """Load a single-rule file.

    Raises if the file holds more than one document, rather than silently
    returning the first and dropping the rest.
    """
    path = Path(path)
    documents = _read_documents(path)
    if len(documents) > 1:
        raise RuleError(
            f"{path}: contains {len(documents)} rule documents; "
            "use load_rules() to load a multi-rule file"
        )
    return parse_rule(documents[0], path=path)


def load_rules(target: str | Path) -> list[Rule]:
    """Load every rule from a file (all documents) or a directory tree.

    Still fail-fast: a broken rule must never be skipped, because a skipped
    rule looks exactly like a rule that passed.  What changed is *when* we
    stop -- errors from every file are collected before raising, so a user
    with twenty rules and three typos fixes all three in one pass instead of
    discover-fix-rerun, three times over.
    """
    path = Path(target)
    if path.is_file():
        return [parse_rule(d, path=path) for d in _read_documents(path)]
    if not path.exists():
        raise RuleError(f"rule path not found: {path}")

    rules: list[Rule] = []
    problems: list[str] = []
    for candidate in sorted(path.rglob("*.y*ml")):
        try:
            documents = _read_documents(candidate)
        except RuleError as exc:
            problems.append(str(exc))
            continue
        for document in documents:
            try:
                rules.append(parse_rule(document, path=candidate))
            except RuleError as exc:
                problems.append(str(exc))

    if problems:
        header = (
            f"{len(problems)} 条规则无法加载（**没有**跳过它们继续跑——"
            "被跳过的规则和通过的规则看起来是一样的）："
        )
        raise RuleError(header + "\n  - " + "\n  - ".join(problems))

    if not rules:
        raise RuleError(f"no rule files found under {path}")
    return rules


def dump_rules(rules: list[Rule]) -> str:
    """Render several rules into one multi-document YAML stream."""
    return "---\n".join(dump_rule(rule) for rule in rules)


def validate_rules(
    rules: list[Rule],
    *,
    enforce_severity: tuple[str, ...] = ("critical", "warning"),
) -> list[str]:
    """Static checks for rules that cannot behave the way the author expects.

    Returns human-readable warnings rather than raising: a mis-scoped rule is
    usually a mistake worth surfacing, not a reason to refuse to start.  These
    are the failure modes that are otherwise completely silent -- the rule
    loads, CI passes, and the guard never fires.
    """
    warnings: list[str] = []
    seen: dict[str, int] = {}

    for rule in rules:
        seen[rule.name] = seen.get(rule.name, 0) + 1

        if rule.is_runtime:
            if rule.severity not in enforce_severity:
                warnings.append(
                    f"规则 {rule.name!r} 的 severity={rule.severity!r} 不在 "
                    f"enforce_severity={sorted(enforce_severity)} 中："
                    "它会被加载但永远不会在运行时生效。"
                )
            if rule.when.turn_range is not None:
                warnings.append(
                    f"规则 {rule.name!r} 使用 `when.turn_range`，而运行时护栏没有对话上下文；"
                    "它实际上只会在 CI 生效。若自行跟踪轮次，请调用 check(turn=<n>)。"
                )
            if rule.then.memory_persistence:
                warnings.append(
                    f"规则 {rule.name!r} 使用 `memory_persistence`，它需要整段对话，"
                    "无法在运行时逐次调用中执行；建议改为 where.runtime: false。"
                )
            if rule.then.require_tool:
                warnings.append(
                    f"规则 {rule.name!r} 使用 `require_tool`，它描述的是整段轨迹的完整性，"
                    "单次工具调用无法判断，运行时不会生效。"
                )
            # `max_tool_calls` was missing from this list while `require_tool`
            # -- equally unanswerable from one call -- was covered.  A budget
            # rule with `where.runtime: true` therefore loaded and validated
            # with no warning, and never fired: the user believes there is an
            # online call budget when there is none.  Same failure class the
            # other three warnings exist to prevent, so it belongs here.
            if rule.then.max_tool_calls is not None:
                warnings.append(
                    f"规则 {rule.name!r} 使用 `max_tool_calls`，它统计的是整段轨迹的调用次数，"
                    "运行时护栏只看得到单次调用，因此不会生效；"
                    "建议改为 where.runtime: false，或自行累计次数后在业务侧判断。"
                )
            # A runtime rule whose every assertion needs the whole trace is a
            # runtime rule that guards nothing.  `guarded_tools()` comes back
            # empty, which the CLI reads as "no rule mentions this tool" -- so
            # the user sees an unguarded ALLOW and cannot tell it apart from a
            # rule that genuinely checked and passed.
            has_runtime_assertion = bool(
                rule.then.forbid_tool or rule.then.allow_tool or rule.then.idempotent
            )
            if not has_runtime_assertion:
                warnings.append(
                    f"规则 {rule.name!r} 标记了 where.runtime: true，但它的断言"
                    "（memory_persistence / require_tool / max_tool_calls）都需要整段轨迹，"
                    "运行时一个都执行不了：这条规则在线上等于不存在。"
                    "若只想在 CI 生效，请改为 where.runtime: false。"
                )
        if not rule.is_ci and not rule.is_runtime:
            warnings.append(
                f"规则 {rule.name!r} 的 where 里 ci 和 runtime 都是 false：它永远不会执行。"
            )

        # An idempotency guard is a safety property, not a scenario-scoped one:
        # charging the same order twice is a bug regardless of what the user
        # typed.  But `when` gates the *whole* rule, so an `idempotent` block
        # nested under `when.intent_any` silently stops policing every turn
        # whose text does not match -- which is exactly when the duplicate
        # happens.  This one bit us for real: with the shipped
        # `rules/refund-safety.yaml`, `--record` wrote the ledger entry and the
        # second call with the same order_id still came back ALLOW.
        if rule.then.idempotent and not rule.when.always:
            gate = []
            if rule.when.intent_any:
                gate.append(f"intent_any={list(rule.when.intent_any)}")
            if rule.when.turn_range is not None:
                gate.append(f"turn_range={list(rule.when.turn_range)}")
            warnings.append(
                f"规则 {rule.name!r} 把 `idempotent` 放在了 `when`（{', '.join(gate)}）之下："
                "只有 when 命中时幂等检查才会跑。重复扣款无论用户说了什么都是事故，"
                "建议把 idempotent 拆成一条不带 when 的独立规则，或至少确认这是你要的语义。"
            )

        for spec in rule.then.memory_persistence:
            too_early = [t for t in spec.must_recall_at if t < spec.inject_at]
            if too_early:
                warnings.append(
                    f"规则 {rule.name!r} 的 memory_persistence 要求在第 {too_early} 轮"
                    f"回忆事实 {spec.fact!r}，但它到第 {spec.inject_at} 轮才注入："
                    "这些检查点必然失败。"
                )
            if spec.must_recall_at and not spec.fact.strip():
                warnings.append(
                    f"规则 {rule.name!r} 的 memory_persistence 事实为空字符串，检查无意义。"
                )

        # `forbid_tool` + `allow_tool` in one rule is the trap the learner
        # already dodges (`learn.py` explains why it emits only `forbid_tool`).
        # Hand-written rules have no such protection, and the failure is
        # silent: the rule loads, the gate runs, and the *reason* reported for
        # a bad call is "name not in the allow-list" rather than "this tool is
        # forbidden" -- so the message no longer matches the author's intent.
        # It also crosses naming systems: the shipped `refund-safety` allowed
        # only English names, so `examples/refund_zh` (Chinese tool names) got
        # four violations that had nothing to do with what the rule meant.
        forbidden_set = set(rule.then.forbid_tool)
        allowed_set = set(rule.then.allow_tool)
        if forbidden_set and allowed_set:
            overlap = sorted(forbidden_set & allowed_set)
            if overlap:
                warnings.append(
                    f"规则 {rule.name!r} 同时出现在 forbid_tool 和 allow_tool 里："
                    f"{overlap}。同一工具既允许又禁止，实际以 forbid 为准，"
                    "请删掉其中一处。"
                )
            covered = sorted(forbidden_set - allowed_set)
            if covered:
                warnings.append(
                    f"规则 {rule.name!r} 的 forbid_tool {covered} 不在 "
                    f"allow_tool {sorted(allowed_set)} 里——allow_tool 是闭名单，"
                    "它已经拒绝了所有未列名的工具，forbid 只是冗余。"
                    "更重要的是：闭名单只认名字写死的工具，"
                    "换一套工具命名（多语言、多 agent）就会把正常调用判成违规。"
                    "想表达「别调用某个工具」，用 forbid_tool 就够了。"
                )

    for name, count in seen.items():
        if count > 1:
            warnings.append(f"规则名 {name!r} 出现 {count} 次，建议改名以便区分报告。")

    warnings.extend(_duplicate_assertion_warnings(rules))
    return warnings


def _duplicate_assertion_warnings(rules: list[Rule]) -> list[str]:
    """Flag assertions that two different rules both make.

    The documented `learn` workflow is "derive, review, commit", and the
    realistic moment to run it is right after an incident -- by which time
    hand-written rules covering the same ground usually already exist.  The
    learner has no way to know that, so it emits a second rule saying exactly
    what `payment-idempotency` already says.  Measured on the bundled incident:
    violation count went from 3 to 6 for one duplicate charge.

    Duplicated assertions are not a correctness bug -- the call is still
    refused -- but they corrupt the *signal*.  One incident becomes two alerts,
    the gate's count no longer tracks the number of problems, a baseline
    recorded before the duplication stops meaning anything, and the operator's
    rational response is to mute one of the rules.  This project's whole claim
    is that the rule file is the asset, so quietly doubling it is worth
    surfacing.
    """
    warnings: list[str] = []

    # (kind, identifier) -> rules asserting it.
    owners: dict[tuple[str, str], list[str]] = {}

    def claim(kind: str, identifier: str, rule_name: str) -> None:
        owners.setdefault((kind, identifier), []).append(rule_name)

    for rule in rules:
        for tool in rule.then.forbid_tool:
            claim("forbid_tool", tool, rule.name)
        for tool in rule.then.allow_tool:
            claim("allow_tool", tool, rule.name)
        for spec in rule.then.idempotent:
            claim("idempotent", f"{spec.tool}|{spec.key}", rule.name)

    _LABEL = {
        "forbid_tool": "forbid_tool {id}",
        "allow_tool": "allow_tool {id}",
        "idempotent": "幂等约束 {id}",
    }

    for (kind, identifier), rule_names in sorted(owners.items()):
        unique = list(dict.fromkeys(rule_names))
        if len(unique) < 2:
            continue
        shown = {
            "forbid_tool": f"forbid_tool {identifier!r}",
            "allow_tool": f"allow_tool {identifier!r}",
            "idempotent": f"幂等约束 {identifier!r}",
        }[kind]
        warnings.append(
            f"{len(unique)} 条规则都在声明同一个 {shown}：{unique}。"
            "它们都会生效，因此同一个工具调用会被报告多次——一次事故变成多条违规，"
            "gate 的数量不再等于问题的数量，之前记下的 baseline 也会失效。"
            "若这是 `learn` 推导出来的规则，说明它已经存在于手写规则里，"
            "直接删掉学到的这一条即可。"
        )

    return warnings


def dump_rule(rule: Rule) -> str:
    """Render a rule back to YAML (used by ``learn``)."""
    _require_yaml()
    then: dict[str, Any] = {}
    if rule.then.forbid_tool:
        then["forbid_tool"] = list(rule.then.forbid_tool)
    if rule.then.allow_tool:
        then["allow_tool"] = list(rule.then.allow_tool)
    if rule.then.require_tool:
        then["require_tool"] = list(rule.then.require_tool)
    if rule.then.max_tool_calls is not None:
        then["max_tool_calls"] = rule.then.max_tool_calls
    if rule.then.idempotent:
        then["idempotent"] = [
            {"tool": item.tool, "key": item.key} for item in rule.then.idempotent
        ]
    if rule.then.memory_persistence:
        then["memory_persistence"] = [
            {
                "fact": m.fact,
                "inject_at": m.inject_at,
                "must_recall_at": list(m.must_recall_at),
            }
            for m in rule.then.memory_persistence
        ]

    document: dict[str, Any] = {"version": rule.version, "rule": rule.name}
    if rule.description:
        document["description"] = rule.description
    if rule.source_trace:
        document["source_trace"] = rule.source_trace
    if not rule.when.always:
        when: dict[str, Any] = {}
        if rule.when.intent_any:
            when["intent_any"] = list(rule.when.intent_any)
        if rule.when.turn_range:
            when["turn_range"] = list(rule.when.turn_range)
        document["when"] = when
    document["then"] = then
    document["where"] = {"ci": rule.where.ci, "runtime": rule.where.runtime}
    document["severity"] = rule.severity
    return yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
