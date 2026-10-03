# AgentGuard

> **一份规则，两个执行出口。**
> 把线上事故变成一条约束，这条约束既在 CI 里卡住回归，也在生产里拦住下一次调用。

[![Tests](https://img.shields.io/badge/tests-302%20passing-brightgreen)](#测试)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#安装)

**🌐 [English](README.en.md) · 简体中文**

---

## 问题

Agent 出事故的标准剧本：

```
周一：改了 system prompt，感觉效果更好了 → 上线
周三：客服反馈 Agent 开始乱调工具 → 回滚
周五：发现问题出在 prompt 改动的第 3 天，但 trace 已经滚掉了
```

现在有两条路，都不通：

- **测试框架**（promptfoo / DeepEval）能在 CI 里断言工具调用，但**测试完就完了**，线上照旧出事。
- **运行时护栏**（NeMo Guardrails）能在工具执行前拦截，但**和你的 CI 测试是两套东西**，规则各写一遍，还会漂移。

结果是同一件事付两遍成本。AgentGuard 把这两条路接到同一份规则上。

---

## 核心主张

```yaml
# rules/refund-safety.yaml —— 场景相关的禁止，用 when 圈定范围
rule: refund-safety
when:
  intent_any: ["物流", "订单", "status"]

then:
  forbid_tool: refund_order          # 禁止调用

where:
  ci: true                           # ← 在 CI 里当断言
  runtime: true                      # ← 在生产里当护栏
```

```yaml
# rules/payment-idempotency.yaml —— 同一订单不得重复退款/扣款
rule: payment-idempotency

# 刻意不写 when：`when` 会把整条规则门控掉。
# 重复扣款无论用户说了什么都是事故，所以这条必须无条件生效。
then:
  idempotent:
    tool: refund_order
    key: "${order_id}"

where:
  ci: true
  runtime: true
```

> **为什么幂等要单独一条规则？** 因为 `when` 门控的是**整条规则**，不是单条断言。
> 把 `idempotent` 写在 `when` 之下，只有触发词命中的那些轮次才做幂等检查 ——
> 而重复扣款恰恰常发生在用户没再说触发词的轮次。
> `validate_rules` 会检测这个写法并告警（这个坑我们真的踩过）。

> **为什么禁用工具时不写 `allow_tool`？** 因为它是**闭名单**：运行时它会拒绝所有
> 未列名的工具。只在单条英文 trace 上试，它看起来没问题；一旦换一套工具命名
> （中文工具名、另一个 agent、多语言），正常调用就会被判成违规 —— 而且报告写的
> 是「名字不在名单里」，跟你想表达的「只读场景别退款」已经不是一回事了。
> 这个坑我们踩过：`examples/refund_zh` 用中文工具名，被英文 `allow_tool`
> 判出 4 条假阳性。**想禁某个工具，`forbid_tool` 就够了**，
> `validate_rules` 会在两者并存时告警。

**`where` 是唯一的开关。** 同一份 YAML：

| 出口 | 时机 | 行为 |
|---|---|---|
| **CI 门禁** | 发布前回放 trace | 违规 → exit 1 → 阻断合并 |
| **Runtime Guard** | 工具调用前 | 违规 → 抛 `ToolBlocked`，**函数根本不会执行** |

---

## 快速开始

```bash
pip install -e .

agentguard init                    # 生成 rules/ + traces/（干净基线）+ examples/（事故样例）
agentguard run                     # 基线通过 → 门禁本身是好的
agentguard run --traces examples   # 事故被拦 → 门禁真的能拦

agentguard guard --rules rules --tool refund_order --arg order_id=A1001 --user "我的订单到哪了"
agentguard guard --rules rules --store ledger.json --record \
                 --tool charge_card --arg order_id=A1001 --user "我要付款"
```

跑完整演示（含事故复现、规则推导、CI 拦截、运行时拦截、生态导出）：

```bash
python examples/demo.py
```

### 目录约定

| 路径 | 内容 |
|---|---|
| `rules/` | 规则（CI 与运行时共用） |
| `traces/` | **干净基线**——规则对它必须全部通过 |
| `examples/` | 演示用的**事故** trace（会触发违规） |

这个划分很重要：`agentguard run`（不带参数）默认跑 `rules/` + `traces/`。
**基线必须是绿的**，否则你无法区分「门禁坏了」和「真的抓到了回归」——
后者是门禁最容易失去信任的失败模式。`traces/baseline-clean.json` 由测试守护。

`agentguard init` 会一次生成这三个目录，并明确告诉你各自的用途：

```
wrote: rules\refund-safety.yaml
wrote: rules\payment-idempotency.yaml
wrote: traces\baseline-clean.json      干净基线 —— 规则对它应当全部通过
wrote: examples\refund_incident.json   事故现场 —— 规则命中它，演示门禁能拦人
```

模板刻意把幂等拆成**独立一条不带 `when` 的规则** —— 原因见上面的「核心主张」。

> **别把 `examples/` 指给 CI。** 里面的 `refund_incident/trace.json` 是**事故现场**，
> 它被规则命中才是正确的。本仓库的 CI 因此跑两个 job：一个跑 `pytest`（验证门禁本身没坏），
> 一个对 `traces/` 跑 `agentguard gate`（验证基线仍然干净）。想看门禁拦人，跑 `examples/demo.py`。

---

## 六类断言

全部**确定性**，不依赖 LLM judge —— 这是刻意的：会误报的门禁，一周内就会被关掉。

| 断言 | 抓什么 | 生态现状 |
|---|---|---|
| `forbid_tool` | 调了不该调的工具 | promptfoo 有等价能力 |
| `allow_tool` | 调了白名单之外的工具 | promptfoo 无 allow-list 原语 |
| `require_tool` | 该调的没调 | promptfoo 有 |
| `max_tool_calls` | 疑似无限循环 | promptfoo 有 |
| **`idempotent`** | **同一事务重复执行（退款/扣款）** | **无开源工具提供** |
| **`memory_persistence`** | **第 N 轮忘了第 M 轮的约束** | **无开源工具提供** |

后两类是 AgentGuard 相对现有生态的真实增量：

- **`idempotent`** 需要知道*哪个参数标识一笔事务*——这是业务语义，通用 eval 框架够不着。
- **`memory_persistence`** 是 **per-turn** 断言。promptfoo 只有整段对话的 `conversation-relevance`，没有「第 20 轮必须还记得第 2 轮的事实」这个原语。

---

## 命令

| 命令 | 作用 |
|---|---|
| `agentguard init` | 生成骨架 |
| `agentguard learn --trace X --apply` | **从 trace 推导候选规则**（人工确认后写入） |
| `agentguard run` | 回放 trace，输出报告，返回退出码 |
| `agentguard gate` | 同 `run`，面向 CI |
| `agentguard guard` | 询问运行时护栏：这次调用能否执行 |
| `agentguard rules` | 列出规则及其守住的工具 |
| `agentguard export --to promptfoo\|nemo` | 导出为其它生态的配置 |

### `guard`：`--record` 决定幂等账本写不写

CLI 的 `guard` 模拟一次工具调用，但**默认只问「能不能调」，不写账本**——
这和真实集成里的 `check()` → 执行 → `record()` 两步是对应的。所以：

```bash
agentguard guard --store ledger.json --record --tool charge_card --arg order_id=A1001
```

- **加 `--record`**：放行后写入账本，**下次同键调用会被拦**。验证幂等必须加它。
- **不加**：只查询，账本（和 `--store` 指定的文件）都不会被创建。

> 这条很容易踩：不加 `--record` 时 `idempotent` 永远不触发，看起来像功能坏了。
> 所以 CLI 会在放行时主动提示「未写入幂等账本」。

### `guard --history`：让运行时看到整段会话

`when.intent_any` 在两个执行器里读的**不是同一段文本**，这是个必须显式处理的差异：

| | 意图匹配的范围 |
|---|---|
| CI | **整条会话**的所有用户输入（`Trace.user_text()`） |
| 运行时 | 默认**只有当前这一轮** |

所以下面这个会话，CI 会拦、运行时（不给 `--history`）会放：

```
第 1 轮  我想查订单物流
第 2 轮  算了，帮我查下天气      → 调用 refund_order
```

CI 拦下（会话里出现过「订单」），运行时只看第 2 轮 → 放行。**危险调用只要不和触发词同轮，就能绕过运行时护栏。**

把前序轮次用 `--history` 交给它，两个执行器就对齐了：

```bash
agentguard guard --rules rules --tool refund_order --arg order_id='"A1"' \
  --user "算了，帮我查下天气" \
  --history "我想查订单物流"
# → [BLOCK] 工具被规则 'refund-safety' 禁止
```

**不传 `--history` 时会主动告警**（而不是给你一个看起来很干净的 ALLOW），
因为那可能是「缺上下文」而不是「真的安全」。

> 运行时**不可能**和 CI 完全一致：它看不到还没发生的轮次。
> 因此不变量是**单向**的 —— 只要触发词在**当前及之前**的轮次里出现过，
> 运行时就必须拦。`test_runtime_cannot_be_bypassed_by_dropping_the_trigger_word`
> 会对 `rules/` + `examples/` 里的真实数据逐条验证这一点。

### `gate --format sarif`：把违规送进 GitHub Security 标签页

```bash
agentguard gate --rules rules --traces examples --format sarif > results.sarif
```

- `--format` 支持 `terminal` / `markdown` / `json` / `sarif`。
- `--out <dir>` 会一次写出 `agentguard-results.json`、`agentguard-report.md`、`agentguard-results.sarif`。
- 严重度映射：`critical → error`、`warning → warning`、`info → note`。
- 规则描述会进 SARIF 的规则元数据（GitHub 告警的说明面板）。
- `properties` 里带 `kind` / `severity` / `turn` / `tool` 和规范英文的 `canonicalMessage`，
  可以和运行时日志对上。

**一条诚实的限制**：结果定位到**文件级**，不是行级。我们知道 trace 文件和轮次，
但不知道那个轮次在 JSON 里的字节位置（`json.loads` 会丢掉位置信息）。
所以 SARIF 里**没有 `region`**，轮次写在消息里。行级定位需要一个带位置信息的 trace 解析器 ——
编造一个 `startLine` 只会把标记画到随便一行上。

### `learn`：不是「一键魔法」

```bash
$ agentguard learn --trace examples/refund_incident/trace.json

  [ 90%] #########  refund-incident-abc123-idempotent-refund_order — refund_order 用同一 order_id 被调用了多次
        · turns 4 & 5 share order_id='A1001'
  [ 82%] ########   refund-incident-abc123-forbid-refund_order — 检测到只读意图的会话中调用了破坏性工具

预览模式：2 条候选达到阈值。确认无误后加 --apply 写入
（AgentGuard 不会自动应用规则——判断哪一步是错的，必须由人确认。）
```

**为什么必须人工确认**：判断「哪一步是错的」是语义问题，启发式会误判。所以 `learn` 只输出**候选 + 置信度 + 证据**，`--apply` 才落盘。任何声称「一键把 trace 变成测试」的方案，都在回避这个难点。

### `when` 里的意图词是从对话里**提取**的，不是整句照抄

`intent_any` 是**子串匹配**，所以拿整句用户原话当条件是没用的——它永远不会再出现一次，
规则等于白学。中文没有词边界，`learn` 因此按停用字切分再取关键词：

```
我上周那单还没到，帮我看下物流   →  intent_any: [物流]
对，就是那个蓝色的              →  intent_any: [蓝色]
```

**切不出关键词时会主动降级**：置信度压到 0.6 以下（`--apply` 默认不写入），
并在生成规则的 `description` 里写明「本规则没有 `when`，会对所有场景生效，请手动补充」。
因为空的 `when` 意味着**无条件生效**——那会把合法的退款/扣款一起拦掉。

### 工具词表可扩展（中英文内置）

`learn` 靠**工具名**判断哪些是破坏性操作。内置了中英文常见词表，并支持追加项目特有名称：

```bash
agentguard learn --trace examples/refund_zh/trace.json \
  --hints docs/hints.example.yaml
```

```yaml
# docs/hints.example.yaml —— 与内置词表合并
destructive: [核销, 冻结, 调价, 补发]
readonly:    [预览, 试算]
```

中文工具名开箱可用：

```
$ agentguard learn --trace examples/refund_zh/trace.json
  [82%] 退款事故-中文工具名-forbid-退款 — 检测到只读意图的会话中调用了破坏性工具 ['退款']
  [70%] 退款事故-中文工具名-idempotent-退款 — 退款 用同一 订单号 被调用了多次
```

> **只读优先**：`查询退款` 会被判为只读（它是退款状态的读取者，不是执行者）。

---

## 不跟生态打架：当翻译层

AgentGuard **不打算成为下一个标准**。promptfoo 已经占住了 CI 断言执行，NeMo Guardrails 已经占住了运行时拦截。跟它们竞争是输的。

所以我们做**翻译层**——你维护一份规则，我们生成你已经在用的工具的原生配置：

```bash
$ agentguard export --to promptfoo
tests:
- description: '[agentguard] refund-safety'
  assert:
  - type: not-trajectory:tool-used      # ← 即「禁止调用该工具」
    value: refund_order

$ agentguard export --to nemo
rails:
  tool_output:
    flows: [tool call validation]       # ← 必须是内置名，见下
```

> `not-trajectory:tool-used` 的可行性是**读 promptfoo 源码验证过的**
> （`src/assertions/trajectory.ts:49 applyInverse()`），不是猜的。

### 导出是「尽力而为」，缺口会明说

不是所有约束都能翻译过去，**导出时会逐条列出翻不了的部分**，而不是生成看起来对、实际不触发的配置：

| 约束 | promptfoo | NeMo Guardrails |
|---|---|---|
| `forbid_tool` | ✅ `not-trajectory:tool-used` | ❌ 无禁止名单原语 |
| `require_tool` | ✅ `trajectory:tool-used` | ❌ |
| `max_tool_calls` | ✅ `trajectory:step-count` | ❌ |
| `allow_tool` | ❌ 无 allow-list 原语 | ✅ 声明为 allowed tools |
| `idempotent` | ❌ 无跨调用账本 | ❌ |
| `memory_persistence` | ❌ 无 per-turn 断言 | ❌ |
| **`when`（意图条件）** | ❌ 无条件断言原语 | ❌ 全局名单，无法按会话切换 |

> **`when` 丢失是最隐蔽的一条**：导出的断言会变成**无条件**生效。
> 一条本该只在物流场景禁止退款的规则，导到 promptfoo 后会变成**全局禁止退款**——
> 合法的退款测试全部失败，你会以为工具坏了。所以导出时会明确告警。

**两个坑，都验证过**：

- **NeMo 的 `rails.tool_output.flows` 只接受内置名 `tool call validation`。**
  用户不写自定义 Colang；**流程名写错会让 NeMo 静默回退到 LLMRails 并禁用全部工具护栏** ——
  你请求的护栏会悄无声息地消失。本工具因此**不生成任何 Colang**。
- NeMo 工具护栏还需要：IORAILS 引擎（`NEMO_GUARDRAILS_IORAILS_ENGINE=1`）、
  OpenAI Chat Completions 线格式、Colang 1.0。导出时会一并提示。

---

## 集成到你的 Agent

```python
from agentguard import RuntimeGuard, FileStore, ToolBlocked

guard = RuntimeGuard.from_rules("rules/", store=FileStore("ledger.json"))

for warning in guard.warnings:      # 加载时就告诉你哪些规则不会生效
    log.warning(warning)

def dispatch(name: str, args: dict):
    # 一行包装，现有的工具分发即可接入
    return guard.execute(name, args, lambda: TOOLS[name](**args), user=current_user_input)
```

被拦时抛 `ToolBlocked`，业务函数**不会被执行**。

**异步 Agent**（LangGraph、Agents SDK 等）用 `aexecute` / `acheck`，签名完全对称：

```python
async def dispatch(name: str, args: dict):
    return await guard.aexecute(
        name, args, lambda: TOOLS[name](**args),
        user=current_user_input,
        history=prior_user_turns,        # 前序轮次，见上文
    )
```

**`history` 请务必传**：它是「前序轮次的用户输入」列表（最早的在前）。
不传时 `when.intent_any` 只看当前这一轮，与 CI 的整会话匹配不一致。

```python
decision = guard.check(
    "refund_order", {"order_id": "A1001"},
    user=current_user_input,
    history=["我想查订单物流"],   # 第 1 轮
)
```

### 拦截原因：`code` 给程序看，`reason` 给日志看

`GateDecision` 同时带机器可读的 `code` 和规范英文的 `reason`：

```python
decision = guard.check("refund_order", {"order_id": "A1001"}, user=text)

decision.allowed   # False
decision.code      # 'forbidden_tool'  ← 用来分支，别去匹配文案
decision.reason    # "tool 'refund_order' is forbidden by rule 'refund-safety'"
decision.detail    # {'tool': 'refund_order'}
```

| `code` | 含义 |
|---|---|
| `allowed` | 放行 |
| `forbidden_tool` | 命中 `forbid_tool` |
| `tool_not_allowed` | 不在 `allow_tool` 内（「除名单外全拒」） |
| `duplicate_transaction` | 幂等账本里已有同一事务 |

**`reason` 故意保持英文** —— 它要进生产日志、CI 的 JSON 和 `ToolBlocked` 的堆栈，
稳定 ASCII 更好检索、也更好贴给同事。**只有终端输出和 Markdown 报告会渲染中文**，
靠的就是 `code` / `kind` 这两个标识符。

要在**自己的日志或告警**里用中文，调 `decision_zh`（已从顶层导出），不要自己去匹配 `reason` 的文案：

```python
from agentguard import decision_zh

try:
    guard.execute(name, args, do_it, user=text)
except ToolBlocked as exc:
    # code 用来分支，decision_zh 用来给人看
    log.warning("%s | %s", exc.decision.code, decision_zh(
        code=exc.decision.code,
        reason=exc.decision.reason,
        detail=exc.decision.detail,
        rule=exc.decision.rule,
    ))
    # → duplicate_transaction | 重复事务：该 order_id='A1001' 已经执行过一次（规则 'payment-idempotency'）
```

CI 侧的违规行同理由 `violation_zh(kind=..., message=..., detail=..., turn=..., tool=...)` 渲染。

> 新增断言时如果忘了加中文渲染，`test_every_violation_kind_has_a_chinese_rendering`
> 会直接从 `engine.py` 里扫出新的 `kind` 并让测试失败 —— 不会静默漏掉。

### 运行时到底执行哪些约束

不是所有断言都能在运行时生效——这取决于它需不需要「整段对话」：

| 断言 | 运行时 | 原因 |
|---|---|---|
| `forbid_tool` | ✅ | 单次调用即可判断 |
| `allow_tool` | ✅ | 单次调用即可判断（见下方说明） |
| `idempotent` | ✅ | 靠幂等账本跨调用判断 |
| `require_tool` | ❌ | 描述整条轨迹的完整性，单次调用无从判断 |
| `memory_persistence` | ❌ | 需要整段对话 |
| `when.intent_any` | ⚠️ | 需要把前序轮次传进 `history`，否则只看当前轮 |
| `when.turn_range` | ⚠️ | 需要轮次上下文，默认拿不到 |

**加载时会对这些情况主动告警**，不会静默失效（`agentguard rules` 也会列出）。

> **`allow_tool` 在运行时是个「大锤」**：它意味着*除名单外一律拒绝*。
> 这是与 CI 保持一致的正确语义（同一份规则必须得到同一结论），
> 但它影响面大，所以提供 `RuntimeGuard(..., enforce_allow_list=False)` 关闭。
> 默认开启——因为「CI 拦、运行时放」正是本项目要消灭的漂移。
>
> 用它的时候记住：**名单是穷举的，而工具命名体系不是**。如果你只想表达
> 「别调用某个工具」，写 `forbid_tool`，不要写 `allow_tool`——后者会把未来
> 新增的工具、换名后的工具、别的语言命名的工具全部判成违规。
> 一条规则里 `forbid_tool` 不写进自己的 `allow_tool` 时，`validate_rules` 会告警。

---

## 设计取舍

| 决策 | 理由 |
|---|---|
| **纯 Python，零依赖树**（除 PyYAML） | 需要装一堆依赖的门禁，不会有人装进 CI |
| **不做 LLM judge 参与门禁** | judge 会误报；误报 → 团队不信任 → 门禁被关。语义检查可以有，但永不单独卡人 |
| **不上数据库** | 规则进 git，报告进 CI artifact，够用 |
| **不做 Web UI（一期）** | 战场在 CI 和 CLI |

---

## 测试

```bash
pip install -e ".[dev]"
pytest -q          # 302 passed
```

覆盖：六类断言的正反用例、规则解析的畸形输入、两个执行器的隔离（`where` 语义）、幂等账本的跨进程持久化、导出格式、CLI 端到端。

---

## 已知限制

诚实列出，而不是等你踩到：

**执行能力**
- **`require_tool` / `memory_persistence` 无法在运行时执行**，只能进 CI。加载时会告警，不会静默失效。
- **`allow_tool` 在运行时是「除名单外全拒」**，影响面大；可用 `enforce_allow_list=False` 关闭。
- **`severity="info"` 的规则默认不在运行时执行**（`enforce_severity` 默认只含 critical/warning）。
- **`when.intent_any` 在运行时只看你交给它的文本**：不传 `history` 就只有当前这一轮，
  而 CI 看整条会话。所以危险调用若发生在触发词之后的轮次，**CI 拦、运行时会放** ——
  CLI 在这种情况下会主动告警，但库调用方需自己保证传 `history`。
  运行时也不可能与 CI 完全一致（它看不到未来的轮次），因此不变量是单向的。
- **`when.turn_range` 只决定规则「是否激活」，不限制它「检查哪些调用」**。
  一条 `turn_range: [1,1]` + `forbid_tool: X` 的规则，会拦下第 9 轮的 `X`。
  CI 一直如此（`rule_applies` 只做激活判断，`_check_forbid` 扫描全轨迹）。
  这是已知的语义疙瘩，`test_turn_range_gates_activation_not_scope` 把它钉住了。
- **`when` 门控的是整条规则，不是单条断言。** 所以把 `idempotent` 写在 `when` 之下，
  只有触发词命中的轮次才做幂等检查 —— 重复扣款恰恰常发生在用户没再说触发词的轮次。
  **幂等属于无条件安全属性，请拆成一条不带 `when` 的规则。**
  写成门控形式时 `validate_rules` 会告警，`rules/` 与 `init` 模板都已按此拆开。
- **一次工具调用可能同时违反多条断言，运行时只回报优先级最高的那条**（`code`），
  其余放在 `detail["also_violated"]`。优先级：`duplicate_transaction` >
  `forbidden_tool` > `tool_not_allowed` —— 因为「这笔钱已经扣过两次」比
  「这个工具不允许」更值得让人先看到。**关键是拦截结论保持一致**，只是理由取了最重要的一条。
- **`guard` 对一个没有任何规则提到的工具会返回 ALLOW**，但会明确提示
  「只是无话可说，不等于已确认安全」并列出被守住的工具 —— 否则打错工具名看起来就像「安全」。

**导出能力**（`agentguard export`）
- **只有 `forbid_tool` / `require_tool` / `max_tool_calls` 能翻译成 promptfoo 断言。**
  `allow_tool`、`idempotent`、`memory_persistence` 在 promptfoo 里没有对应原语——
  导出时会**明确列出并告警**，而不是生成看起来正确、实际不会触发的断言。
- 导出的 promptfoo 配置里 `providers` 是占位符，需要你按自己的 trace 回放方式补齐。

**解析与启发式**
- **规则文件里的未知键会被拒绝**（含拼写建议），不再静默忽略。
  这条很重要：`when.intent` 这种拼错的键若被忽略，`when` 会变空 —— 而空的 `when` 意味着
  **无条件生效**，与作者本意正好相反。
- **规则目录里的错误会一次报全，但仍然拒绝加载**。收集所有文件的错误再抛出（一轮修完），
  绝不跳过坏规则继续跑 —— 被跳过的规则和通过的规则看起来是一样的。
- **读不到名字的工具调用不会消失**。`tool_calls` 里缺 `name`/`tool`/`tool_name` 的条目会被计为
  `<unnamed-tool-call>`，并在报告里带 `[!]` 告警。丢弃它们会让「我没读懂这条 trace」
  和「这条 trace 没有工具调用」显示成同一个绿色的 PASS。
- **`learn` 的启发式会误判**，所以强制人工确认。
- **`memory_persistence` 是字面量匹配**，不是语义比对。有意为之（避免 judge 误报），但换个说法就测不出来。
- **`when.intent_any` 是整条 trace 的子串匹配**，粒度为「整个会话」而非「某一轮」。
  一个规则只要会话里出现过关键词就会整体生效。运行时通过 `history` 对齐这个粒度。

**工程**
- **幂等是「先占位、后执行」**：`execute` 在调用业务函数**之前**就原子占住账本键，
  业务成功则保留、抛异常则回滚。这一点很重要 —— 早先的实现在 `check` 与 `record`
  之间让业务函数整个跑完，那个窗口里到达的并发调用都看到空账本：
  实测 12 个并发对同一 `order_id` 扣款，**12 个全部放行**。现在同一场景是
  「1 次执行 + 11 次 `duplicate_transaction`」。失败回滚的语义是：
  业务没成功就不算一次事务，重试合法；一旦成功，同键重复即被拦。
- **自定义存储后端的接口要求**：实现 `seen(key)` / `record(key)` 即可正常工作；
  再实现 `reserve(key) -> bool` / `release(key)` 才能获得并发保护。
  只实现前两个不会报错，但 `RuntimeGuard.warnings` 里会出现一条降级告警 ——
  并发保护会静默丢失，所以必须说出来。内置的 `MemoryStore` / `FileStore` 都实现了完整四个方法。
- **`gate --baseline` 的输入格式**：同时接受 `{"summary": {"violations": N}}`（即
  `run --format json` 的输出，可直接重定向存档）和扁平的 `{"violations": N}`。
  **文件存在但读不出违规数时会直接报错并退出 1**，不会当作「没有 baseline」跳过比较 ——
  把损坏的 baseline 当成「无需比较」会让回归门禁静默失效。文件不存在则视为尚未采用基线，正常放行。
- **`FileStore` 写入是原子的**（临时文件 + `os.replace`），并且**读-改-写全程持有跨进程文件锁**
  （`O_CREAT|O_EXCL`，带超时破锁），因此多进程并发预占同一键也只会有一个赢家。
  锁只在一次小 JSON 写入期间持有；持锁进程若被强杀，超过 10s 后锁会被自动打破。
  高并发场景仍建议换成 Redis/DB 适配器（文件锁每次操作都有一次磁盘往返，不适合高 QPS）。
- **账本读不出来时会直接报错，不会当作空账本继续跑**。把损坏内容当成「还没执行过任何事务」
  会让所有幂等检查静默失效 —— 那正是这个工具要防的事故。文件不存在或为空字节视为全新开始。
- **`RuntimeGuard` 提供同步与异步两套入口**（`check`/`execute` 与 `acheck`/`aexecute`），
  异步版只是 `await` 业务函数，判定逻辑与同步版完全共用。
- **没有 OTel 摄入**：`load_trace` 只读 JSON/JSONL 文件，不会主动连 OTel collector。
- **SARIF 结果是文件级的，没有行号**：`json.loads` 丢掉了位置信息，所以 SARIF 里不写 `region`，
  轮次写在消息里。行级定位需要带位置信息的 trace 解析器。
- **CI 回放的确定性依赖 trace 质量**：模型非确定性是根本难题，
  本项目通过「门禁只由确定性断言驱动」绕开它，而不是解决它。
- **导出到 promptfoo 时 `when` 门控会丢失**：promptfoo 没有「仅当…时」的条件断言原语，
  导出的断言会在所有场景生效，比原规则更严格。该约束记在
  `metadata.agentguard_when.ported: false` 里，并在 stderr、文件头注释、
  `metadata.agentguard_unsupported` 三处告警。
- **`learn` 会重复推导已有规则**：最自然的执行时机是事故之后，那时手写规则通常已经在管同一件事。
  `learn` 现在会在写盘前提示「这条已被 X 覆盖」（可用 `--rules` 指定要比对的规则），
  `validate_rules` 也会对**完全相同**的重复断言告警 —— 重复断言仍会生效，
  会让一次事故报出多条违规，使「违规数」不再等于「问题数」，并使既有 baseline 失真。

**测试**
- 302 个测试覆盖断言正反例、规则解析畸形输入、两个执行器的一致性、
  多文档 YAML 往返、幂等账本跨进程持久化与跨进程原子性、导出缺口标注、中文工具词表、SARIF 结构、
  异步入口、报告中的 trace 归属、真实 trace 格式（OpenAI / OTel / Langfuse 风格）、CLI 端到端。
- 其中 `tests/test_regressions.py` 集中存放**审计中实际发现的缺陷**，每个 bug 一个回归测试。
- 还有几条**结构性守护**，防止已修的坑以别的形式回来：测试函数不得重复定义
  （重复定义会让先出现的版本静默不执行）、`rules/` 里不得把 `idempotent` 藏在 `when` 之下、
  `rules/` 里不得让 `forbid_tool` 被自己的 `allow_tool` 覆盖（闭名单会误伤别的工具命名体系）、
  `learn` 生成的 `when.intent_any` 不得含破坏性词（否则「用户要退款」会触发「禁止退款」）、
  导出配置里不得存在无人消费的 `vars`（会让 `when` 门控看起来被迁移了）、
  `__all__` 不得漂移。
- **未覆盖**：真实 LLM 调用、Windows 之外的路径行为。

---

## License

Apache-2.0
