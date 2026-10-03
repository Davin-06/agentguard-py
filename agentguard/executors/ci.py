"""CI executor -- the post-hoc half of "one rule, two executors".

Two things matter here and nowhere else in the ecosystem:

1. **Determinism.**  A gate that is flaky gets switched off within a week.
   So we never call a live model in the gate path; the trace is a recorded
   script and every check in :mod:`agentguard.engine` is deterministic.
2. **Severity thresholds.**  Not every finding should block a deploy.
   ``--fail-on critical`` is the default; ``--fail-on warning`` is stricter;
   ``--fail-on none`` turns the run into a pure report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..engine import evaluate_all
from ..models import SEVERITY_ORDER, Trace
from ..report import RunResult, render_markdown, render_sarif, render_terminal, to_dict
from ..rules import Rule, load_rules
from ..trace import load_traces


@dataclass
class GateOutcome:
    runs: RunResult
    exit_code: int
    threshold: str
    notes: list[str] = field(default_factory=list)
    #: The CI rules that were actually evaluated.  Carried so renderers can
    #: attach rule descriptions (SARIF help panels) without re-reading the
    #: rule files.
    rules: list[Rule] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.exit_code == 0

    def blocking_violations(self) -> list[Any]:
        floor = SEVERITY_ORDER.get(self.threshold, 99)
        return [
            v
            for _trace, results in self.runs
            for r in results
            for v in r.violations
            if SEVERITY_ORDER.get(v.severity, 0) >= floor
        ]

    def total_violations(self) -> list[Any]:
        """Every violation, regardless of the current threshold.

        Baseline comparison must use this, not :meth:`blocking_violations`,
        otherwise changing ``--fail-on`` between runs makes the comparison
        meaningless.
        """
        return [
            v for _trace, results in self.runs for r in results for v in r.violations
        ]


def run_gate(
    rules_path: str | Path,
    traces_path: str | Path,
    *,
    fail_on: str = "critical",
    baseline: str | Path | None = None,
) -> GateOutcome:
    rules: list[Rule] = load_rules(rules_path)
    traces: list[Trace] = load_traces(traces_path)

    # Only rules marked `where.ci: true` participate in the CI gate.
    ci_rules = [r for r in rules if r.is_ci]
    runs: RunResult = [(trace, evaluate_all(ci_rules, trace)) for trace in traces]

    exit_code = 0
    if fail_on != "none":
        floor = SEVERITY_ORDER.get(fail_on, 2)
        for _trace, results in runs:
            for result in results:
                for v in result.violations:
                    if SEVERITY_ORDER.get(v.severity, 0) >= floor:
                        exit_code = 1
                        break

    outcome = GateOutcome(
        runs=runs, exit_code=exit_code, threshold=fail_on, rules=ci_rules
    )

    if baseline is not None:
        note = _check_baseline(baseline, outcome)
        if note:
            outcome.notes.append(note)
            outcome.exit_code = 1
    return outcome


class BaselineError(RuntimeError):
    """A baseline file exists but cannot be used.

    Treating an unreadable baseline as "no baseline" silently disables the
    regression check -- the gate goes green and the team believes a recorded
    floor is being enforced when nothing is compared at all.  That is the same
    failure mode as reading a corrupt ledger as empty, and it gets the same
    answer: refuse to run rather than run blind.
    """


def _read_baseline_count(path: Path) -> int | None:
    """Extract the recorded violation count, or raise :class:`BaselineError`.

    Two accepted spellings, because both are things a user will plausibly write:

    * ``{"summary": {"violations": N}}`` -- the shape ``--format json`` emits, so
      ``agentguard run --format json > baseline.json`` round-trips.
    * ``{"violations": N}`` -- the obvious flat form.  This used to be ignored in
      silence: the number was there, the comparison never happened, and a real
      regression on a clean baseline was never reported.  Accepting it costs
      nothing; refusing to guess would be the wrong trade for a safety input.

    Returns ``None`` only when the file has no violation count in either form --
    which the caller turns into a hard error, since a baseline without a count is
    a mistake, not an instruction to skip the check.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BaselineError(f"baseline {path} 存在但无法读取：{exc}") from exc

    if not raw_text.strip():
        raise BaselineError(
            f"baseline {path} 是空文件。空 baseline 无法参与比较，"
            "要么删掉它（表示这次不做回归比较），要么写入一个违规数。"
        )

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise BaselineError(
            f"baseline {path} 不是合法 JSON（{exc}）。"
            "继续运行会让回归检查静默失效，因此已中止。"
        ) from exc

    if not isinstance(data, dict):
        raise BaselineError(f"baseline {path} 应为 JSON 对象，实际是 {type(data).__name__}")

    if isinstance(data.get("violations"), int) and not isinstance(data["violations"], bool):
        return data["violations"]

    summary = data.get("summary")
    if isinstance(summary, dict):
        value = summary.get("violations")
        if isinstance(value, int) and not isinstance(value, bool):
            return value

    raise BaselineError(
        f"baseline {path} 里找不到违规数（需要 summary.violations 或 violations）。"
        f"实际键：{sorted(data)}。"
        "缺字段的 baseline 会让回归检查静默失效，因此已中止。"
    )


def _check_baseline(baseline: str | Path, outcome: GateOutcome) -> str | None:
    """Fail if the *total* violation count got worse than a recorded baseline.

    Compares like with like: both sides count every violation, not just the
    ones above the current threshold.  Mixing "total" from the baseline file
    with "blocking only" from this run produced false regressions whenever
    ``--fail-on`` differed between runs.

    A *missing* file returns ``None`` -- that is a project which has not adopted
    a baseline yet.  A file that exists but cannot be read raises, because that
    is a configuration error and silently skipping the comparison would hide a
    regression.
    """
    path = Path(baseline)
    if not path.exists():
        return None

    previous = _read_baseline_count(path)
    current = len(outcome.total_violations())
    if current > previous:
        return (
            f"violations regressed: {previous} -> {current} "
            f"（baseline {path}）"
        )
    return None


def write_reports(outcome: GateOutcome, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out / "agentguard-results.json",
        "markdown": out / "agentguard-report.md",
        # SARIF belongs in the artifact bundle too: GitHub's upload-sarif action
        # then finds it without a second command.
        "sarif": out / "agentguard-results.sarif",
    }
    paths["json"].write_text(
        json.dumps(to_dict(outcome.runs), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["markdown"].write_text(render_markdown(outcome.runs), encoding="utf-8")
    paths["sarif"].write_text(
        render_sarif(outcome.runs, outcome.rules), encoding="utf-8"
    )
    return paths


def print_report(outcome: GateOutcome) -> None:
    print(render_terminal(outcome.runs))
