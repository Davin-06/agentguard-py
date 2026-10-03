<div align="center">

# AgentGuard

**One rule. Two enforcement points.**

Turn a production incident into a single constraint — that constraint then
blocks the regression in CI *and* blocks the next offending call in production.

[![Tests](https://img.shields.io/badge/tests-302%20passing-brightgreen)](#tests)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#install)
[![Zero deps](https://img.shields.io/badge/runtime%20deps-1%20(PyYAML)-informational)](#design-trade-offs)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)

[中文文档](README.md) · [Quick start](#quick-start) · [Why](#the-problem)

</div>

---

## The problem

The standard script for an agent incident:

```
Mon  Changed the system prompt. Looks better. Ship it.
Wed  Support reports the agent is calling the wrong tools. Roll back.
Fri  Traced it to day 3 of the prompt change — but the traces already rotated out.
```

There are two existing paths, and neither closes the loop:

- **Test frameworks** (promptfoo, DeepEval) can assert on tool calls in CI —
  but once the test passes, nothing stops production from doing it anyway.
- **Runtime guardrails** (NeMo Guardrails) can block before a tool executes —
  but they're a *separate* set of rules from your CI tests, so the two drift apart.

You end up paying for the same safety property twice. AgentGuard wires both
enforcement points to the **same rule file**.

---

## The core idea

```yaml
# rules/refund-safety.yaml — scope a prohibition to a scenario with `when`
rule: refund-safety
when:
  intent_any: ["shipping", "order", "status"]

then:
  forbid_tool: refund_order

where:
  ci: true                           # ← enforced as an assertion in CI
  runtime: true                      # ← enforced as a guardrail in production
```

**`where` is the only switch.** The same YAML file:

| Enforcement point | When | Behavior |
|---|---|---|
| **CI gate** | Replays traces before release | Violation → `exit 1` → blocks the merge |
| **Runtime guard** | Before the tool call | Violation → raises `ToolBlocked`, **the function never runs** |

---

## Quick start

```bash
pip install -e .

agentguard init                     # scaffolds rules/ + traces/ + examples/
agentguard run                      # the clean baseline passes — the gate itself works
agentguard run --traces examples    # the incident is caught — the gate actually catches things
```

Run the full demo (incident reproduction → rule inference → CI block → runtime block → ecosystem export):

```bash
python examples/demo.py
```

### Directory conventions

| Path | Contents |
|---|---|
| `rules/` | Rules — shared by CI and runtime |
| `traces/` | **Clean baseline** — rules must pass against these |
| `examples/` | Incident traces used for demos (these *should* trip violations) |

This split matters. `agentguard run` with no arguments defaults to `rules/` + `traces/`.
**The baseline must stay green** — otherwise you can't tell "the gate broke" from
"we actually caught a regression," which is the fastest way for a gate to lose the
team's trust.

> **Don't point CI at `examples/`.** Those are *incident scenes*; being caught by the
> rules is the correct outcome for them. This repo's own CI runs two jobs: one runs
> `pytest` (verifying the gate itself isn't broken), one runs `agentguard gate`
> against `traces/` (verifying the baseline is still clean).

---

## What's actually new here

### Six assertion types — all deterministic, no LLM judge

This is deliberate: **a gate that cries wolf gets turned off within a week.**

| Assertion | Catches | Ecosystem status |
|---|---|---|
| `forbid_tool` | Calling a tool it shouldn't | promptfoo has an equivalent |
| `allow_tool` | Calling a tool outside the allow-list | **No allow-list primitive in promptfoo** |
| `require_tool` | Failing to call a required tool | promptfoo has it |
| `max_tool_calls` | Suspected infinite loops | promptfoo has it |
| **`idempotent`** | **The same transaction running twice (refund / charge)** | **No open-source tool provides this** |
| **`memory_persistence`** | **Forgetting a turn-2 constraint by turn 20** | **No open-source tool provides this** |

The last two are the real delta:

- **`idempotent`** requires knowing *which argument identifies a transaction* —
  that's business semantics, out of reach for a generic eval framework.
- **`memory_persistence`** is a **per-turn** assertion. promptfoo only has
  conversation-level `conversation-relevance`; it has no primitive for
  "by turn 20 it must still remember the fact from turn 2."

### Multi-turn state drift

Most evals test a single response. AgentGuard replays long sessions and asserts
that constraints *persist*:

```yaml
- memory_persistence:
    fact: "user budget cap is 5000"   # injected at turn 2
    must_recall_at: [5, 10, 20]
```

### Not fighting the ecosystem — acting as a translation layer

AgentGuard **does not intend to become the next standard.** promptfoo already owns
CI assertion execution; NeMo Guardrails already owns runtime interception. Competing
with them is a losing move.

So we're a **translation layer** — you maintain one rule, we emit native config for
the tools you already run:

```bash
$ agentguard export --to promptfoo
tests:
- description: '[agentguard] refund-safety'
  assert:
  - type: not-trajectory:tool-used      # ← i.e. "forbid this tool"
    value: refund_order
```

> The feasibility of `not-trajectory:tool-used` was **verified by reading promptfoo's
> source** (`src/assertions/trajectory.ts:49 applyInverse()`), not guessed.

Exports are **best-effort, and the gaps are stated explicitly** rather than emitting
config that looks right but never fires. For example, `when` (the intent condition)
has no equivalent in promptfoo — so an exported assertion becomes **unconditional**,
and the tool warns you loudly at export time.

---

## Integrate with your agent

```python
from agentguard import RuntimeGuard, FileStore, ToolBlocked

guard = RuntimeGuard.from_rules("rules/", store=FileStore("ledger.json"))

for warning in guard.warnings:      # tells you at load time which rules won't fire
    log.warning(warning)

def dispatch(name: str, args: dict):
    # one-line wrapper; your existing tool dispatch plugs straight in
    return guard.execute(name, args, lambda: TOOLS[name](**args), user=current_user_input)
```

When blocked, `ToolBlocked` is raised and **your business function never executes**.

Async agents (LangGraph, Agents SDK, …) use `aexecute` / `acheck` — identical signatures:

```python
async def dispatch(name: str, args: dict):
    return await guard.aexecute(
        name, args, lambda: TOOLS[name](**args),
        user=current_user_input,
        history=prior_user_turns,        # prior turns — see below
    )
```

**Always pass `history`**: it's the list of prior user inputs (oldest first). Without it,
`when.intent_any` only sees the current turn in runtime, which does not match CI's
whole-session matching.

### Machine-readable block reasons

`GateDecision` carries both a machine-readable `code` and a canonical English `reason`:

```python
decision = guard.check("refund_order", {"order_id": "A1001"}, user=text)

decision.allowed   # False
decision.code      # 'forbidden_tool'   ← branch on this, never on the prose
decision.reason    # "tool 'refund_order' is forbidden by rule 'refund-safety'"
```

| `code` | Meaning |
|---|---|
| `allowed` | Passed |
| `forbidden_tool` | Hit `forbid_tool` |
| `tool_not_allowed` | Not in `allow_tool` ("deny everything else") |
| `duplicate_transaction` | The idempotency ledger already holds this transaction |

### What actually runs at runtime

Not every assertion can run at runtime — it depends on whether it needs the whole conversation:

| Assertion | Runtime | Why |
|---|---|---|
| `forbid_tool` | ✅ | Decidable from a single call |
| `allow_tool` | ✅ | Decidable from a single call |
| `idempotent` | ✅ | Decided via the idempotency ledger across calls |
| `require_tool` | ❌ | Describes whole-trajectory completeness |
| `memory_persistence` | ❌ | Needs the whole conversation |
| `when.intent_any` | ⚠️ | Needs prior turns in `history`, else only the current turn |
| `when.turn_range` | ⚠️ | Needs turn context |

**These cases warn at load time** — they never fail silently.

---

## Design trade-offs

| Decision | Rationale |
|---|---|
| **Pure Python, zero dependency tree** (except PyYAML) | A gate that needs a dependency tree won't get installed into CI |
| **No LLM judge in the gate** | Judges produce false positives; false positives → distrust → the gate gets turned off |
| **No database** | Rules live in git, reports live in CI artifacts. That's enough |
| **No web UI (v1)** | The battlefield is CI and the CLI |

---

## Commands

| Command | Purpose |
|---|---|
| `agentguard init` | Scaffold `rules/` + `traces/` + `examples/` |
| `agentguard learn --trace X --apply` | **Infer candidate rules from a trace** (human confirmation required) |
| `agentguard run` | Replay traces, emit a report, return an exit code |
| `agentguard gate` | Same as `run`, but CI-facing |
| `agentguard guard` | Ask the runtime guardrail: can this call execute? |
| `agentguard rules` | List rules and the tools they cover |
| `agentguard export --to promptfoo\|nemo` | Emit config for other ecosystems |

### `gate --format sarif` — violations into the GitHub Security tab

```bash
agentguard gate --rules rules --traces examples --format sarif > results.sarif
```

**One honest limitation**: results are located at **file level, not line level**.
We know the trace file and the turn, but not the byte offset of that turn inside the
JSON (`json.loads` discards position info). So the SARIF has **no `region`** — the turn
is written into the message. Fabricating a `startLine` would just paint the marker on
a random line.

### `learn` — not "one-click magic"

```bash
$ agentguard learn --trace examples/refund_incident/trace.json

  [ 90%] #########  refund-incident-abc123-idempotent-refund_order
        · turns 4 & 5 share order_id='A1001'
  [ 82%] ########   refund-incident-abc123-forbid-refund_order
        · destructive tool called in a read-only-intent session

Preview mode: 2 candidates above threshold. Add --apply to write.
```

**Why human confirmation is mandatory**: deciding *which step was wrong* is a semantic
problem, and heuristics misjudge it. Any tool claiming "one-click trace → test" is
dodging that hard part.

---

## Tests

```bash
pip install -e ".[dev]"
pytest -q          # 302 passed
```

Coverage includes positive/negative cases for all six assertions, malformed rule input,
isolation between the two enforcement points (`where` semantics), cross-process
idempotency-ledger durability, export formats, and CLI end-to-end.

`tests/test_regressions.py` holds **defects actually found during audits** — one
regression test per bug.

---

## Known limitations

Stated honestly rather than left for you to discover:

**Enforcement**
- **`require_tool` / `memory_persistence` cannot run at runtime** — CI only. They warn at load time.
- **`allow_tool` is a sledgehammer at runtime**: it means *deny everything not listed*.
  Disable with `RuntimeGuard(..., enforce_allow_list=False)`.
- **Rules with `severity="info"` don't run at runtime by default.**
- **`when.intent_any` at runtime only sees the text you hand it**: without `history`,
  a dangerous call in a later turn is caught by CI but passes at runtime. The CLI warns
  about this; library callers must pass `history` themselves.
- **`when` gates the whole rule, not a single assertion.** Idempotency is an
  unconditional safety property — split it into its own rule without `when`.

**Export**
- **Only `forbid_tool` / `require_tool` / `max_tool_calls` translate to promptfoo assertions.**
  `allow_tool`, `idempotent`, `memory_persistence` have no promptfoo primitive — export
  **lists and warns about them explicitly**.

**Engineering**
- **Idempotency is "reserve first, execute second"**: `execute` atomically claims the
  ledger key *before* calling your business function. An earlier implementation let the
  business function fully complete between `check` and `record` — measured result:
  **12 concurrent charges against the same `order_id`, all 12 passed.** The same scenario
  now yields "1 execution + 11 `duplicate_transaction`."
- **`FileStore` writes are atomic** (temp file + `os.replace`) and hold a cross-process
  file lock for the read-modify-write. High-concurrency deployments should swap in a
  Redis/DB adapter.
- **No OTel ingestion**: `load_trace` reads JSON/JSONL files only.
- **SARIF results are file-level, no line numbers** (see above).

---

## License

Apache-2.0

---

<div align="center">

**Your traces tell you what happened. AgentGuard tells you whether you can ship.**

</div>
