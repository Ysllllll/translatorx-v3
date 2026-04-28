"""Chapter 2 — Pipeline bypass mechanisms (TranslateProcessor + Store, no real LLM)."""

from __future__ import annotations

import json as _json
from dataclasses import replace as _replace  # noqa: F401  (kept for symmetry)
from pathlib import Path
from tempfile import TemporaryDirectory

from adapters.storage.store import JsonFileStore
from adapters.storage.workspace import Workspace
from application.checker import CheckReport, Checker
from application.processors import TranslateProcessor
from application.processors.prefix import TranslateNodeConfig
from application.terminology import StaticTerms
from application.translate import TranslationContext
from application.translate.variant import VariantSpec
from domain.model import SentenceRecord
from domain.model.usage import CompletionResult, Usage
from ports.source import VideoKey

from ._common import header, sub, truncate


class _FakeTranslator:
    """Deterministic engine — counts calls to verify bypass behaviour."""

    def __init__(self, tag: str = "fake-zh-v1") -> None:
        self.calls = 0
        self.model = tag
        self.log: list[tuple[str, str]] = []

    async def complete(self, messages, **_):
        self.calls += 1
        src = messages[-1]["content"]
        out = f"【译·{self.model}】{src}"
        self.log.append((src, out))
        return CompletionResult(text=out, usage=Usage())

    async def stream(self, messages, **_):
        yield (await self.complete(messages)).text


class _PassChecker(Checker):
    def __init__(self) -> None:
        super().__init__(rules=[])

    def check(self, _src, _tgt, _profile=None, **_):
        return CheckReport(issues=[])


def _ctx(variant: VariantSpec | None = None) -> TranslationContext:
    return TranslationContext(
        source_lang="en",
        target_lang="zh",
        terms_provider=StaticTerms({}),
        window_size=4,
        variant=variant or VariantSpec(),
    )


def _rec(rid: int, text: str, **extra):
    base = {"id": rid, **extra}
    return SentenceRecord(src_text=text, start=float(rid), end=float(rid) + 1.0, extra=base)


async def _exec(proc, recs, store, vkey, *, variant: VariantSpec | None = None):
    async def src():
        for r in recs:
            yield r

    out = []
    async for r in proc.process(src(), ctx=_ctx(variant), store=store, video_key=vkey):
        out.append(r)
    return out


def _classify(rec: SentenceRecord) -> str:
    tgt = rec.get_translation("zh") or ""
    if not tgt:
        return "EMPTY"
    if tgt.startswith("【译·"):
        return "LLM "
    if tgt == rec.src_text:
        return "SKIP"
    return "DIRECT"


async def run() -> None:
    header("Chapter 2 — 流水线旁路机制（真实 TranslateProcessor + Store）")
    print(
        "  本节完整跑通 runtime 层:  Workspace → JsonFileStore → TranslateProcessor。\n"
        "  三条旁路（direct_translate / fingerprint-cache / max_source_len）\n"
        "  都用真实 processor.process() 走一遍，并把源文本混合成 direct+LLM+\n"
        "  skip 的复杂组合，便于观察 per-record 命中分布。\n"
        "  跑完之后打印 workspace 目录树 + zzz_translation/*.json 原始内容。"
    )

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        ws = Workspace(root=root, course="ml_101")
        store = JsonFileStore(ws)

        # ─── 2a direct_translate 混合 LLM 兜底 ────────────────────────────
        sub("2a  direct_translate 字典命中 + LLM 兜底（混合场景）")
        print("    6 条句子：4 条命中字典 → 0 LLM call；\n    2 条未命中 → 正常走 LLM。验证字典只是短路逻辑，不是全封顶。")
        vkey_a = VideoKey(course="ml_101", video="lec01_intro")
        eng_a = _FakeTranslator("fake-a")
        cfg_a = TranslateNodeConfig(
            direct_translate={
                "ok": "好的",
                "yeah": "是的",
                "um": "嗯",
                "thanks": "谢谢",
                "welcome back": "欢迎回来",
            }
        )
        proc_a = TranslateProcessor(eng_a, _PassChecker(), config=cfg_a)
        inputs_a = [
            _rec(0, "ok"),
            _rec(1, "Thanks"),
            _rec(2, "Today we're talking about gradient descent."),
            _rec(3, "um"),
            _rec(4, "The optimizer minimizes the loss function."),
            _rec(5, "welcome back"),
        ]
        out_a = await _exec(proc_a, inputs_a, store, vkey_a)
        for r in out_a:
            kind = _classify(r)
            print(f"    [{kind}] id={r.extra['id']}  {truncate(r.src_text, 48)!r:52s} → {truncate((r.get_translation('zh') or '∅'), 30)!r}")
        direct_hits = sum(1 for r in out_a if _classify(r) == "DIRECT")
        llm_hits = sum(1 for r in out_a if _classify(r) == "LLM ")
        print(f"    ⇒ direct={direct_hits}  llm={llm_hits}  engine.calls={eng_a.calls}")
        assert direct_hits == 4 and eng_a.calls == 2

        # ─── 2b variant cache：三次 run 观察命中 / 失效 ────────────────
        sub("2b  variant 缓存 — 三次 run 观察命中 → 命中 → 失效重算")
        print(
            "    Run1: 全新 store → 全部走 LLM，写入 translations[zh][variant_key]\n"
            "    Run2: 同 variant 同记录 → 缓存命中 → 全部跳过 LLM\n"
            "    Run3: 换 variant.alias → variant_key 变化 → 全部重翻"
        )
        vkey_b = VideoKey(course="ml_101", video="lec02_optimizers")
        eng_b = _FakeTranslator("fake-b")
        cfg_b = TranslateNodeConfig(system_prompt="You translate ML tutorials.")
        proc_b = TranslateProcessor(eng_b, _PassChecker(), config=cfg_b)
        variant_v1 = VariantSpec(alias="v1")
        variant_v2 = VariantSpec(alias="v2-concise")
        inputs_b = [
            _rec(0, "SGD uses a single mini-batch per step."),
            _rec(1, "Momentum accumulates past gradients."),
            _rec(2, "Adam combines momentum with RMSProp."),
            _rec(3, "Weight decay is L2 regularization."),
            _rec(4, "The learning rate schedule matters a lot."),
        ]

        n_before = eng_b.calls
        await _exec(proc_b, inputs_b, store, vkey_b, variant=variant_v1)
        run1_delta = eng_b.calls - n_before
        print(f"    Run1  engine.calls +{run1_delta}  (全部新翻，应为 5)")

        n_before = eng_b.calls
        out_b2 = await _exec(proc_b, inputs_b, store, vkey_b, variant=variant_v1)
        run2_delta = eng_b.calls - n_before
        cached_zh = [(r.get_translation("zh") or "") for r in out_b2]
        print(f"    Run2  engine.calls +{run2_delta}  (variant_key 命中，应为 0)")
        print(f"          cached[0] = {truncate(cached_zh[0], 60)!r}")
        assert run2_delta == 0

        assert variant_v1.key != variant_v2.key
        n_before = eng_b.calls
        await _exec(proc_b, inputs_b, store, vkey_b, variant=variant_v2)
        run3_delta = eng_b.calls - n_before
        print(f"    Run3  engine.calls +{run3_delta}  (variant_key 失效，应为 5)")
        assert run3_delta == 5

        # ─── 2c max_source_len skip + direct + LLM 三者混合 ────────────────
        sub("2c  max_source_len skip + direct_translate + LLM（三路混合）")
        print(
            "    5 条字幕：1 direct 命中、2 普通长度走 LLM、2 超长被 skip。\n    skip 的 translation 字段应保留原文或空，方便后续人工修正。"
        )
        vkey_c = VideoKey(course="ml_101", video="lec03_messy_asr")
        eng_c = _FakeTranslator("fake-c")
        long_a = (
            "Sometimes the ASR glues many sentences together without "
            "punctuation and we get this absurdly long run-on text which "
            "is basically unusable for LLM translation because attention "
            "gets diluted and the model drops half the content entirely."
        )
        long_b = (
            "Another pathological transcript where forty seconds of speech "
            "collapses into one paragraph due to a broken VAD threshold, "
            "with three different speakers overlapping and the whisper "
            "decoder hallucinating extra clauses that were never spoken."
        )
        cfg_c = TranslateNodeConfig(
            max_source_len=120,
            direct_translate={"hello.": "你好。"},
        )
        proc_c = TranslateProcessor(eng_c, _PassChecker(), config=cfg_c)
        inputs_c = [
            _rec(0, "Hello."),
            _rec(1, "The loss decreased steadily."),
            _rec(2, long_a),
            _rec(3, "We trained for 100 epochs."),
            _rec(4, long_b),
        ]
        out_c = await _exec(proc_c, inputs_c, store, vkey_c)
        for r in out_c:
            kind = _classify(r)
            L = len(r.src_text)
            print(f"    [{kind}] id={r.extra['id']}  len={L:>3d}  → {truncate((r.get_translation('zh') or '∅'), 55)!r}")
        print(f"    ⇒ engine.calls = {eng_c.calls}  (应为 2：两条正常长度的)")
        assert eng_c.calls == 2

        sub("📁  Workspace 目录结构（JsonFileStore 写入后）")
        print(f"    root = {root}")
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            depth = len(rel.parts) - 1
            indent = "    " + "  " * depth
            suffix = "/" if p.is_dir() else ""
            size = f"  ({p.stat().st_size} B)" if p.is_file() else ""
            print(f"{indent}├─ {rel.parts[-1]}{suffix}{size}")

        sub("📜  zzz_translation/*.json 内容（Store 的物理落盘）")
        tx_dir = root / "ml_101" / "zzz_translation"
        for jp in sorted(tx_dir.glob("*.json")):
            print(f"    ── {jp.name} " + "─" * (60 - len(jp.name)))
            data = _json.loads(jp.read_text(encoding="utf-8"))
            meta = data.get("meta", {})
            recs = data.get("records") or data.get("sentences") or []
            print(f"      schema_version: {data.get('schema_version')}")
            print(f"      meta._fingerprints: {meta.get('_fingerprints')}")
            print(f"      records: {len(recs)} 条")
            for r in recs[:2]:
                rid = r.get("id")
                zh = (r.get("translations") or {}).get("zh", "")
                if isinstance(zh, dict):
                    zh = next(iter(zh.values()), "")
                tgt = truncate(zh or "", 50)
                extra = r.get("extra", {})
                print(f"        • id={rid}  zh={tgt!r}")
                if extra:
                    print(f"          extra={extra}")
            if len(recs) > 2:
                print(f"        … +{len(recs) - 2} more records")
