"""``llm_ops`` 完整翻译链路可观测 demo（六章节，从简到繁）。

章节安排：

1. **Checker 规则矩阵** — 纯本地质量规则，不调 LLM。每条内置规则都给出
   一个会命中 ERROR 的样例，外加 hallucination-start、bracket、multi-rule、
   WARNING-only passing 等情况。
2. **流水线旁路机制** — 三种不调 LLM 的旁路：direct_translate 字典、
   fake_process 已有译文复用、max_source_len 超长跳过。
3. **单句真实翻译** — 空 window、无 terms，打印发给 LLM 的 messages。
4. **完整流水线** — PreloadableTerms 预加载 summary/terms + compact frozen
   few-shot + 逐条翻译，每步打印 window BEFORE/AFTER。最后一次调用还会
   打印完整 messages + resolved system prompt。
5. **Prompt 降级** — ScriptedEngine 强制回落，逐级展示 Level 0 → 1 → 2 → 3
   每一级送入 LLM 的 messages 结构差异。Level 3 是新加的 bare 回落。
6. **流式场景** —
   6a. ``OneShotTerms``：逐条 ``request_generation`` 累积到阈值触发后台
       摘要/术语抽取；观察 ``ready`` 从 False 翻到 True。
   6b. ``engine.stream``：实时 yield chunks，边收到边打印。

LLM 端点：``http://localhost:26592/v1``。1/2/5 章节纯本地可跑；3/4/6 章节
在 LLM 不可达时会自动跳过。运行::

    python demos/demo_llm_ops.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  —— 把 src/ 挂到 sys.path

import asyncio
from contextlib import suppress
from dataclasses import dataclass

import httpx

from checker import CheckReport, Checker, Severity, default_checker
from checker.rules import (
    FormatRule,
    KeywordRule,
    LengthRatioRule,
)
from checker.types import Issue
from llm_ops import (
    ContextWindow,
    OneShotTerms,
    PreloadableTerms,
    StaticTerms,
    TranslationContext,
    build_frozen_messages,
    get_default_system_prompt,
    translate_with_verify,
)
from llm_ops.protocol import Message
from model.usage import CompletionResult, Usage
import trx


LLM_BASE_URL = "http://localhost:26592/v1"
LLM_MODEL = "Qwen/Qwen3-32B"

SEP = "═" * 72
SUB = "─" * 72


# ═══════════════════════════════════════════════════════════════════════
# 通用打印工具
# ═══════════════════════════════════════════════════════════════════════

def _truncate(text: str, limit: int = 120) -> str:
    text = text.replace("\n", " ⏎ ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def header(title: str) -> None:
    print("\n" + SEP)
    print(title)
    print(SEP)


def sub(title: str) -> None:
    print("\n" + SUB)
    print(f"  {title}")


def print_messages(messages: list[Message], *, limit: int = 120) -> None:
    """以紧凑形式打印一次 engine.complete 的 messages。"""
    if not messages:
        print("    (空)")
        return
    for i, m in enumerate(messages):
        content = _truncate(m["content"], limit)
        print(f"    [{i:2d}] {m['role']:9s} | {content}")


def print_system_prompt(prompt: str) -> None:
    """按行打印 system prompt — 方便观察 metadata 是否正确注入。"""
    for line in prompt.splitlines():
        print(f"    │ {line}")


def print_window(window: ContextWindow) -> None:
    if len(window) == 0:
        print("    (空)")
        return
    pairs = window.build_messages()
    for i in range(0, len(pairs), 2):
        print(f"    [{i // 2}] src: {_truncate(pairs[i]['content'], 90)}")
        print(f"        tgt: {_truncate(pairs[i + 1]['content'], 90)}")


def print_report(report: CheckReport) -> None:
    status = "✓ passed" if report.passed else "✗ failed"
    if not report.issues:
        print(f"    {status}  (no issues)")
        return
    print(f"    {status}  ({len(report.issues)} issue(s))")
    for iss in report.issues:
        marker = "!" if iss.severity == Severity.ERROR else "·"
        print(f"      {marker} [{iss.severity.value:7s}] {iss.rule}: {iss.message}")


# ═══════════════════════════════════════════════════════════════════════
# 本地 engine — 不需要真实 LLM
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LoggingEngine:
    """装饰真实 engine，拦截 complete() 与 stream() 记录最近一次 messages。"""

    inner: object
    last_messages: list[Message] | None = None

    async def complete(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
        self.last_messages = list(messages)
        return await self.inner.complete(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
        )

    async def stream(self, messages, *, temperature=None, max_tokens=None):
        self.last_messages = list(messages)
        async for chunk in self.inner.stream(
            messages, temperature=temperature, max_tokens=max_tokens,
        ):
            yield chunk


class ScriptedEngine:
    """按脚本依次返回结果的假 engine。"""

    def __init__(self, scripted_replies: list[str]) -> None:
        self._replies = list(scripted_replies)
        self.call_log: list[list[Message]] = []

    async def complete(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
        self.call_log.append(list(messages))
        reply = self._replies.pop(0) if self._replies else "(empty)"
        return CompletionResult(text=reply, usage=Usage())

    async def stream(self, messages, *, temperature=None, max_tokens=None):
        self.call_log.append(list(messages))
        reply = self._replies.pop(0) if self._replies else "(empty)"
        for tok in reply:
            yield tok


# ═══════════════════════════════════════════════════════════════════════
# Section 1 — Checker 规则矩阵（纯本地）
# ═══════════════════════════════════════════════════════════════════════

def section_1_checker_matrix() -> None:
    header("Section 1 — Checker 规则矩阵（不需要 LLM）")
    print(
        "  逐条展示内置规则的命中样例。最后两例展示 ERROR 短路 与 WARNING-only\n"
        "  仍 passed=True 的边界情况。"
    )

    checker = default_checker("en", "zh")
    # 将 length_ratio 配置为 WARNING，用于最后一例：issues 不为空但 passed=True
    warning_ratio = Checker(rules=[LengthRatioRule(severity=Severity.WARNING)])

    cases: list[tuple[str, str, str, Checker | None]] = [
        ("1.1  clean pass",
         "Hello, world.", "你好，世界。", None),
        ("1.2  length_ratio — 译文过长（ratio > 4.0）",
         "Hi.", "你好" * 60 + "。", None),
        ("1.3  format — 幻觉前缀（translation starts with \"好的，这是翻译：\"）",
         "Compute the gradient.",
         "好的，这是翻译：计算梯度。", None),
        ("1.4  format — bracket mismatch（译文以括号开头但源文不是）",
         "See figure a.", "（图a）。", None),
        ("1.5  format — markdown 粗体残留",
         "The loss is minimized.", "**损失** 被最小化。", None),
        ("1.6  format — 意外换行",
         "Step one. Step two.", "第一步。\n第二步。", None),
        ("1.7  question_mark — 源含 \"?\"，译文漏问号（WARNING, 不阻断）",
         "Is this correct?", "这是对的。", None),
        ("1.8  keyword — forbidden 术语命中",
         "We train a model on the data.",
         "我们在数据上狗狗模型。",
         Checker(rules=[KeywordRule(forbidden_terms=["狗狗"])])),
        ("1.9  keyword_pair — 译文幻觉出源文没有的术语",
         "The snake slithered through the grass.",
         "这条 Python 蛇在草丛里滑行。",
         Checker(rules=[KeywordRule(keyword_pairs=[(["python"], ["Python", "python"])])])),
        ("1.10 trailing_annotation — 句末幻觉括号注释",
         "The activation function is ReLU.",
         "激活函数是 ReLU（注：这里指整流线性单元激活函数）。", None),
        ("1.11 多规则 — length + hallucination 都会触发，ERROR 短路",
         "Hi.", "好的，这是翻译：" + "你好" * 50 + "。", None),
        ("1.12 WARNING only — 有 issue 但 report.passed=True",
         "Hello.", "你好" * 30 + "。", warning_ratio),
    ]

    for title, src, tgt, custom in cases:
        sub(title)
        print(f"    source      : {src}")
        print(f"    translation : {_truncate(tgt, 90)}")
        (custom or checker)
        report = (custom or checker).check(src, tgt)
        print_report(report)


# ═══════════════════════════════════════════════════════════════════════
# Section 2 — Direct-translate dict 旁路（纯本地）
# ═══════════════════════════════════════════════════════════════════════

def section_2_direct_translate() -> None:
    header("Section 2 — 流水线旁路机制（不需要 LLM）")
    print(
        "  TranslateProcessor 在调用 LLM 之前会走一系列旁路判断；命中任何\n"
        "  一条都跳过 LLM。这里展示旧 TranslatorX 保留下来的三种旁路：\n"
        "    2a  direct_translate dict — 短语字典命中 → 直接返回\n"
        "    2b  fake_process         — 已有译文（缓存/人工）→ 直接复用\n"
        "    2c  max_source_len skip  — 源文本过长 → 放弃翻译，原样或空串"
    )

    # ── 2a direct_translate
    sub("2a  direct_translate dict")
    direct_translate: dict[str, str] = {
        "ok": "好的",
        "yeah": "是的",
        "um": "嗯",
        "thanks": "谢谢",
        "welcome back": "欢迎回来",
    }
    samples_2a = [
        "ok", "Thanks", "um", "let's train a model",
        "welcome back", "please subscribe", "yeah",
    ]
    for src in samples_2a:
        key = src.strip().lower()
        if key in direct_translate:
            print(f"    ✓ direct  {src!r:30s} → {direct_translate[key]!r}  (LLM skipped)")
        else:
            print(f"    → LLM     {src!r:30s} (fall through to model)")

    # ── 2b fake_process —— 已有译文直接复用
    sub("2b  fake_process — 已存在译文直接复用（缓存/人工复核场景）")
    print(
        "    场景：Store 已有前一轮翻译结果 / 人工复核填好 translation，\n"
        "    TranslateProcessor 检测到 record.translation 非空 → 直接跳过\n"
        "    LLM 调用，但仍然把 (src, tgt) 作为 few-shot 灌入 window，\n"
        "    保持后续新句子的上下文一致性。"
    )
    existing_records = [
        {"src": "Gradient descent minimizes the loss.", "tgt": "梯度下降最小化损失。"},
        {"src": "Adam is an adaptive optimizer.",       "tgt": None},  # 无译文 → 走 LLM
        {"src": "The learning rate is 0.001.",          "tgt": "学习率为 0.001。"},
        {"src": "We regularize with weight decay.",     "tgt": ""},  # 空串 → 走 LLM
    ]
    for rec in existing_records:
        src, tgt = rec["src"], rec["tgt"]
        if tgt:
            print(f"    ✓ fake    {src!r:45s} → {tgt!r}  (window-fed, LLM skipped)")
        else:
            print(f"    → LLM     {src!r:45s} (translation empty, fall through)")

    # ── 2c max_source_len skip
    sub("2c  max_source_len skip — 超长源文本放弃翻译")
    print(
        "    场景：某条字幕被 ASR 误粘到 500+ 字符（常见于切分错误），\n"
        "    LLM 容易漏翻或跑飞。TranslateProcessor 设置 max_source_len\n"
        "    阈值；超过即跳过，返回空串或源文本，交由后续人工修正。"
    )
    max_source_len = 120
    cases_2c = [
        "Hello world.",
        "Gradient descent minimizes the loss function iteratively over many steps.",
        ("Sometimes the ASR system glues many sentences together without "
         "punctuation and we get this absurdly long run-on text which is "
         "basically unusable for LLM translation because attention gets "
         "diluted and the model tends to drop half the content entirely, "
         "so we skip instead of risking a hallucinated shortcut."),
    ]
    for src in cases_2c:
        L = len(src)
        if L > max_source_len:
            print(f"    ⚠ skip    len={L:>4d} > {max_source_len}  {_truncate(src, 70)!r}")
        else:
            print(f"    → LLM     len={L:>4d} ≤ {max_source_len}  {_truncate(src, 70)!r}")


# ═══════════════════════════════════════════════════════════════════════
# Section 3 — 单句真实翻译（最小化场景）
# ═══════════════════════════════════════════════════════════════════════

async def section_3_single_sentence() -> None:
    header("Section 3 — 单句真实翻译（空 window、无 terms）")
    print(
        "  这是最小的真实调用：系统提示默认模板 + 单条 user message。\n"
        "  用来确认 prompt 模板、engine 配置、checker 都 wired 正确。"
    )

    inner = _make_engine()
    engine = LoggingEngine(inner=inner)
    ctx = TranslationContext(source_lang="en", target_lang="zh", max_retries=2)
    checker = default_checker("en", "zh")
    window = ContextWindow(size=6)

    sub("resolved system prompt（无 metadata 注入）")
    print_system_prompt(get_default_system_prompt(ctx))

    source = "Gradient descent minimizes the loss function iteratively."
    sub(f"source: {source}")
    result = await translate_with_verify(source, engine, ctx, checker, window)

    sub("📤 messages sent to LLM")
    print_messages(engine.last_messages or [], limit=110)

    sub("✅ result")
    print(f"    translation : {result.translation}")
    print(f"    attempts={result.attempts}  accepted={result.accepted}")
    print_report(result.report)


# ═══════════════════════════════════════════════════════════════════════
# Section 4 — 完整流水线
# ═══════════════════════════════════════════════════════════════════════

SOURCE_TEXTS: list[str] = [
    "Welcome back everyone, today we are going to talk about reinforcement learning from human feedback, or RLHF.",
    "RLHF is the key ingredient that turned raw large language models into helpful assistants like ChatGPT.",
    "The pipeline has three stages: supervised fine-tuning, reward model training, and policy optimization with PPO.",
    "In the supervised fine-tuning stage, we start from a pretrained language model and fine-tune on curated demonstration data.",
    "The goal of SFT is to teach the model the desired response format before we hand it over to reinforcement learning.",
    "In the reward model stage, we collect pairs of responses and ask human labelers which one is better.",
    "We then train a reward model to predict the human preference, giving a scalar score to any response.",
    "Finally, the policy network is updated to maximize expected reward, with a KL penalty keeping it close to SFT.",
    "The KL penalty is crucial, otherwise the policy drifts off distribution and the reward model stops being accurate.",
    "In practice people often replace PPO with DPO or GRPO because they are simpler and avoid training a separate value network.",
    "Modern variants also add length penalties and safety classifiers directly into the reward signal.",
]


async def section_4_full_pipeline() -> None:
    header("Section 4 — 完整流水线：PreloadableTerms + frozen few-shot + window")
    print(
        "  展示一次完整批翻译的全部可观测面：\n"
        "    • PreloadableTerms 一次性预抽 summary / terms\n"
        "    • build_frozen_messages 将术语压成紧凑 few-shot\n"
        "    • window 随翻译进度滚动\n"
        "    • resolved system prompt 把 topic/field 等 metadata 注入进去\n"
        "    • 最终 messages 打印完整发送内容"
    )

    inner = _make_engine()
    engine = LoggingEngine(inner=inner)

    # ── 4.1 预加载 summary + terms
    sub("4.1  PreloadableTerms 预加载 (6 条源文本)")
    terms = PreloadableTerms(inner, source_lang="en", target_lang="zh")
    await terms.preload(SOURCE_TEXTS)
    print(f"    metadata : {terms.metadata}")
    extracted = await terms.get_terms()
    print(f"    terms    : {len(extracted)} 条")
    for k, v in list(extracted.items())[:8]:
        print(f"      {k!r:45s} → {v!r}")

    ctx = TranslationContext(
        source_lang="en", target_lang="zh",
        terms_provider=terms, window_size=4,
    )
    window = ContextWindow(size=ctx.window_size)
    checker = default_checker("en", "zh")

    # ── 4.2 resolved system prompt（已注入 metadata）
    sub("4.2  resolved system prompt（metadata 已注入 topic/description）")
    print_system_prompt(get_default_system_prompt(ctx))

    # ── 4.3 frozen few-shot（compact primer + 拼接术语对）
    sub("4.3  frozen few-shot（LaTeX primer + 压缩术语对）")
    compact = build_frozen_messages(tuple(extracted.items()))
    print_messages(compact, limit=110)

    # ── 4.4 逐条翻译，每步打印 window BEFORE / AFTER + 本轮完整 messages
    sub("4.4  逐条翻译（每句都打印 BEFORE window / messages / TRANSLATION / AFTER window）")
    for idx, src in enumerate(SOURCE_TEXTS, 1):
        print("\n" + SUB)
        print(f"  #{idx}/{len(SOURCE_TEXTS)}  SRC: {_truncate(src, 90)}")

        print(f"\n  📜 window BEFORE (size={window.size}, 当前={len(window)})")
        print_window(window)

        result = await translate_with_verify(src, engine, ctx, checker, window)

        print(f"\n  📤 messages sent to LLM  (共 {len(engine.last_messages or [])} 条)")
        print_messages(engine.last_messages or [], limit=100)

        print(f"\n  ✅ TRANSLATION: {result.translation}")
        print(f"     attempts={result.attempts} accepted={result.accepted} "
              f"passed={result.report.passed}")

        print(f"\n  📜 window AFTER  (当前={len(window)})")
        print_window(window)

    # ── 4.5 最后一次调用的完整 messages（verbose，不截断）
    sub("4.5  最后一次调用完整 messages（system + frozen + window + user；verbose）")
    print_messages(engine.last_messages or [], limit=200)


# ═══════════════════════════════════════════════════════════════════════
# Section 5 — Prompt 降级（scripted，4 级）
# ═══════════════════════════════════════════════════════════════════════

class _AlwaysFailChecker:
    """Always reject translations — forces translate_with_verify to exhaust retries."""

    def check(self, _src: str, _tgt: str) -> CheckReport:
        return CheckReport(issues=[Issue(
            "demo_force_fail", Severity.ERROR, "demo forcing retry",
        )])


async def section_5_prompt_degradation() -> None:
    header("Section 5 — Prompt 降级（Level 0 / 1 / 2 / 3 bare）")
    print(
        "  用 ScriptedEngine 强制每次都被 checker 判不合格 → 翻译器依次走\n"
        "    Level 0 full        system + frozen few-shot + window + user\n"
        "    Level 1 compressed  history 压进 system（单轮）\n"
        "    Level 2 minimal     system + user（无 history）\n"
        "    Level 3 bare        单条 user，中文兜底指令，无 system\n"
        "  每一级都打印完整 messages，方便肉眼 diff 结构差异。\n"
        "  注意：system_prompt 使用 get_default_system_prompt(ctx) — 与生产一致。"
    )

    source = "Batch normalization stabilizes training across layers."

    replies = ["bad translation"] * 4
    engine = ScriptedEngine(scripted_replies=replies)

    ctx = TranslationContext(
        source_lang="en", target_lang="zh",
        terms_provider=StaticTerms({"gradient descent": "梯度下降"}),
        frozen_pairs=(("gradient descent", "梯度下降"),),
        window_size=4,
        max_retries=3,
    )
    window = ContextWindow(size=4)
    window.add("Gradient descent minimizes the loss.", "梯度下降最小化损失。")
    window.add("Adam is an optimizer.", "Adam 是一个优化器。")

    # 使用生产默认 system prompt（会自动注入 source_lang / target_lang）
    real_prompt = get_default_system_prompt(ctx)
    sub("resolved system prompt（生产默认模板，注入 en → zh metadata）")
    print_system_prompt(real_prompt)

    print(f"\n  SRC       : {source}")
    print(f"  max_retries: {ctx.max_retries}  (共 {ctx.max_retries + 1} 次尝试)")
    print(f"  window primed with 2 pairs, frozen_pairs=1")

    result = await translate_with_verify(
        source, engine, ctx, _AlwaysFailChecker(), window,
        system_prompt=real_prompt,
    )

    level_names = ["L0 full", "L1 compressed", "L2 minimal", "L3 bare"]
    for i, msgs in enumerate(engine.call_log):
        sub(f"attempt #{i + 1} — {level_names[i]}  ({len(msgs)} msg)")
        print_messages(msgs, limit=200)

    sub("最终结果")
    print(f"    translation : {result.translation!r}")
    print(f"    attempts={result.attempts}  accepted={result.accepted}")
    print("    (accepted=False → 该结果不会写入 window)")


# ═══════════════════════════════════════════════════════════════════════
# Section 6a — OneShotTerms 流式术语抽取
# ═══════════════════════════════════════════════════════════════════════

def _describe_task(t) -> str:
    if t is None:
        return "None"
    if t.done():
        if t.cancelled():
            return "cancelled"
        if t.exception() is not None:
            return f"error({type(t.exception()).__name__})"
        return "done"
    return "pending"


def _dump_oneshot_state(terms: OneShotTerms, label: str) -> None:
    """打印 OneShotTerms 的内部观测量 — 全部私有属性，仅用于 demo。"""
    char_cnt = getattr(terms, "_char_count", -1)
    threshold = getattr(terms, "_char_threshold", -1)
    seen = getattr(terms, "_seen_texts", set())
    task = getattr(terms, "_task", None)
    pct = (char_cnt * 100 // threshold) if threshold > 0 else 0
    print(f"    📊 state @ {label}:")
    print(f"       ready          = {terms.ready}")
    print(f"       char_count     = {char_cnt:>4d} / {threshold} ({pct}%)")
    print(f"       seen_texts     = {len(seen)} 条（累计记录，非去重）")
    print(f"       bg_task        = {_describe_task(task)}")


async def section_6a_oneshot_terms() -> None:
    header("Section 6a — OneShotTerms 流式累积（浏览器插件场景）")
    print(
        "  模拟字幕插件：句子逐条到达 → OneShotTerms 累积到 char_threshold 后\n"
        "  后台抽取术语，期间翻译继续；术语就绪后自动并入后续调用。\n"
        "\n"
        "  每一步都打印 OneShotTerms 内部状态：\n"
        "    • ready          — 术语是否已就绪\n"
        "    • char_count     — 已累积字符数 / 阈值\n"
        "    • seen_texts     — 累计收到的文本条数（未去重）\n"
        "    • bg_task        — 后台抽取 Task 的状态 (None / pending / done)"
    )

    engine_inner = _make_engine()
    engine = LoggingEngine(inner=engine_inner)

    stream_texts = [
        "Gradient descent is the workhorse of deep learning.",
        "It updates weights proportional to the loss gradient.",
        "Adam builds on this by adapting the learning rate per parameter.",
        "In practice you tune beta1, beta2 and epsilon.",
        "Batch normalization further stabilizes deep networks.",
        "Dropout regularizes training by randomly masking activations.",
    ]

    terms = OneShotTerms(engine_inner, "en", "zh", char_threshold=100)
    ctx = TranslationContext(
        source_lang="en", target_lang="zh",
        max_retries=2, terms_provider=terms,
    )
    checker = default_checker("en", "zh")
    window = ContextWindow(size=6)

    sub("初始状态（request_generation 之前）")
    _dump_oneshot_state(terms, "init")

    # 演示累积：多次 request 会把字符数持续累计（seen_texts 全量记录，
    # 达到 char_threshold 时后台 Task 才被触发）
    sub("累积演示：两次 request_generation 把 char_count 推近阈值")
    warmup = "This warms up the accumulator toward the threshold."
    await terms.request_generation([warmup])
    _dump_oneshot_state(terms, "after 1st request (warmup)")
    await terms.request_generation([warmup])
    _dump_oneshot_state(terms, "after 2nd request (注意：char_count 会继续累加)")

    for i, src in enumerate(stream_texts, 1):
        sub(f"stream #{i}: {_truncate(src, 80)}")
        _dump_oneshot_state(terms, f"before request #{i}")

        await terms.request_generation([src])
        # 让后台任务（若刚被触发）有调度机会
        await asyncio.sleep(0)
        _dump_oneshot_state(terms, f"after request #{i}  (可能刚触发 bg_task)")

        before = terms.ready
        # 在第 3 条之后显式等待后台抽取完成，从而让第 4 条开始的翻译
        # 能观察到 ready=True 的翻转点。
        if i == 3 and not before:
            print("    ⏳ 等待后台术语抽取完成以观察 ready 翻转 …")
            await terms.wait_until_ready()
            print("    ⚡ READY FLIPPED  (False → True)")
            _dump_oneshot_state(terms, "after wait_until_ready()")
            before = terms.ready

        result = await translate_with_verify(src, engine, ctx, checker, window)
        after = terms.ready
        flip = "  ⚡ READY FLIPPED (during translate)" if (not before and after) else ""
        print(f"    translation: {result.translation}{flip}")

        # 展示本轮发送给 LLM 的 messages — 重点观察:
        #   • before ready 翻转：messages 里不含 frozen few-shot
        #   • after  ready 翻转：messages 开头多出 few-shot pairs（术语对）
        note = "（含 few-shot 术语对）" if after else "（无 terms，仅 system + window + user）"
        print(f"    📤 messages sent to LLM {note}:")
        print_messages(engine.last_messages or [], limit=100)

        if after:
            snap = await terms.get_terms()
            if snap:
                pair_preview = list(snap.items())[:3]
                print(f"    terms so far ({len(snap)}): " + ", ".join(f"{k}→{v}" for k, v in pair_preview) + ("…" if len(snap) > 3 else ""))

    await terms.wait_until_ready()
    sub("final state")
    _dump_oneshot_state(terms, "final")
    final_terms = await terms.get_terms()
    print(f"    terms ({len(final_terms)} 条):")
    for k, v in list(final_terms.items())[:10]:
        print(f"      {k!r:45s} → {v!r}")


# ═══════════════════════════════════════════════════════════════════════
# Section 6b — engine.stream 实时 token
# ═══════════════════════════════════════════════════════════════════════

async def section_6b_engine_stream() -> None:
    header("Section 6b — engine.stream 实时 token 流")
    print("  边收到边打印，统计 chunk 数与总字符数。\n")
    engine = _make_engine(max_tokens=256)

    source = (
        "Reinforcement learning from human feedback has become the standard "
        "recipe for aligning large language models with human preferences."
    )
    print(f"  SRC: {source}\n")
    messages: list[Message] = [
        {"role": "system", "content": "你是英译中意译专家。仅输出中文译文。"},
        {"role": "user", "content": source},
    ]

    print("  streaming:")
    print("    ", end="", flush=True)
    chunks = 0
    total_chars = 0
    async for chunk in engine.stream(messages):
        chunks += 1
        total_chars += len(chunk)
        print(chunk, end="", flush=True)
    print(f"\n\n  ✓ 收到 {chunks} 个 chunk，合计 {total_chars} 字符")


# ═══════════════════════════════════════════════════════════════════════
# Engine factory
# ═══════════════════════════════════════════════════════════════════════

def _make_engine(*, max_tokens: int = 2048):
    return trx.create_engine(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key="EMPTY",
        temperature=0.3,
        max_tokens=max_tokens,
        extra_body={
            "top_k": 20,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

async def _llm_alive() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{LLM_BASE_URL}/models")
            return r.status_code == 200
    except Exception:
        return False


async def main() -> None:
    header(f"demo_llm_ops — 六章节可观测（Sections 1 / 2 / 5 纯本地，3 / 4 / 6 需 LLM）")

    alive = await _llm_alive()
    if alive:
        print(f"\n✅ LLM 在线: {LLM_MODEL} @ {LLM_BASE_URL}")
    else:
        print(f"\n⚠️  LLM 不可达 ({LLM_BASE_URL})，Sections 3 / 4 / 6 将跳过。")

    # 纯本地：永远运行
    section_1_checker_matrix()
    section_2_direct_translate()

    # 需 LLM
    if alive:
        await section_3_single_sentence()
        await section_4_full_pipeline()
    else:
        print("\n(Sections 3 / 4 跳过 — 需要真实 LLM)")

    # 纯本地
    await section_5_prompt_degradation()

    # 需 LLM
    if alive:
        await section_6a_oneshot_terms()
        await section_6b_engine_stream()
    else:
        print("\n(Sections 6a / 6b 跳过 — 需要真实 LLM)")

    header("demo_llm_ops 完成")


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        asyncio.run(main())
