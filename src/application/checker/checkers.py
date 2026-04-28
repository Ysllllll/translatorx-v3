"""Checker — 基于 Scene 的翻译质量检查规则引擎。

两个公开入口：

- :meth:`Checker.check` — 高层便捷方法；接收原始 ``source`` /
  ``target`` 字符串，返回 ``(sanitized_target, report)``。
  调用者无需自行构造 :class:`CheckContext`。
- :meth:`Checker.run`   — 底层方法；接收已构建好的
  :class:`CheckContext`（当调用者需要传入 ``usage`` /
  ``prior`` / ``metadata`` 时使用）。返回 ``(updated_ctx, report)``。

规则可调用对象按 **scene 为单位编译一次**（以 scene 名称为键），
缓存在 Checker 实例上，因此每次调用的开销仅为执行已编译的函数——
无工厂调用、无正则重编译、无字体重加载。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .lang import LangProfile, ScriptFamily, get_profile
from .registry import build as _build_step
from .scene import (
    CheckerConfig,
    SceneConfig,
    _apply_overrides,
    _coerce_rule_entry,
    resolve_scene,
)
from .types import CheckContext, CheckReport, Issue, ResolvedScene, RuleSpec, Severity


@dataclass(frozen=True, slots=True)
class _CompiledScene:
    """已实例化规则可调用对象的已解析 scene。"""

    name: str
    sanitize: tuple[tuple[RuleSpec, Callable[..., str]], ...]
    rules: tuple[tuple[RuleSpec, Callable[..., Any]], ...]


def _compile(resolved: ResolvedScene) -> _CompiledScene:
    sanitize = tuple((spec, _build_step(spec.name, kind="sanitize", **spec.params)) for spec in resolved.sanitize)
    rules = tuple((spec, _build_step(spec.name, kind="check", **spec.params)) for spec in resolved.rules)
    return _CompiledScene(name=resolved.name, sanitize=sanitize, rules=rules)


_EMPTY_METADATA: Mapping[str, Any] = MappingProxyType({})


class Checker:
    """基于 Scene 的翻译质量检查器。"""

    __slots__ = (
        "_source_lang",
        "_target_lang",
        "_scenes",
        "_default_scene",
        "_compiled",
        "_scenes_view",
    )

    def __init__(
        self,
        *,
        source_lang: str = "",
        target_lang: str = "",
        scenes: Mapping[str, SceneConfig] | None = None,
        default_scene: str = "",
    ) -> None:
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._scenes: dict[str, SceneConfig] = dict(scenes or {})
        self._default_scene = default_scene
        # 缓存：scene 名称 → 已编译的可调用对象。
        # 仅在"无覆盖/无规则替换"的热路径上填充。
        # 高级 run() 调用会绕过缓存，即时编译。
        self._compiled: dict[str, _CompiledScene] = {}
        self._scenes_view: Mapping[str, SceneConfig] = MappingProxyType(self._scenes)

    @classmethod
    def from_config(
        cls,
        config: CheckerConfig,
        *,
        source_lang: str = "",
        target_lang: str = "",
    ) -> Checker:
        """根据 :class:`CheckerConfig` 构建 Checker。"""
        return cls(
            source_lang=source_lang,
            target_lang=target_lang,
            scenes=config.scenes,
            default_scene=config.default_scene,
        )

    # ---------------------------------------------------------------- 运行

    def __call__(
        self,
        source: str,
        target: str,
        *,
        usage: Any = None,
        prior: str | None = None,
        scene: str | None = None,
    ) -> tuple[str, CheckReport]:
        """High-level entrypoint — operate on raw strings.

        ``target, report = checker(source, target)``

        Internally builds a :class:`CheckContext` using the Checker's
        bound ``source_lang`` / ``target_lang``, runs sanitize + check,
        and returns ``(sanitized_target, report)``.

        Need ``metadata`` or a different lang pair? Build a fresh
        Checker (``default_checker(src, tgt)``) or call :meth:`run`
        with a hand-built :class:`CheckContext`.
        """
        ctx = CheckContext(
            source=source,
            target=target,
            source_lang=self._source_lang,
            target_lang=self._target_lang,
            usage=usage,
            prior=prior,
        )
        new_ctx, report = self.run(ctx, scene=scene)
        return new_ctx.target, report

    # Explicit alias for callers that prefer named methods over __call__.
    check = __call__

    def run(
        self,
        ctx: CheckContext,
        *,
        scene: str | None = None,
        rules: Sequence[Any] | None = None,
        overrides: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[CheckContext, CheckReport]:
        """在指定 *scene* 下为 *ctx* 运行 sanitize + check 管线。

        解析顺序：

        1. ``scene`` 参数 > ``self._default_scene``。如果两者都未设置，
           抛出 :class:`ValueError`。
        2. ``rules=[...]`` 会 *替换* 已解析的规则列表
           （scene 中的 sanitize 步骤仍会执行）。
        3. ``overrides={name: {severity?, params?}}`` 在其上叠加覆盖。

        返回 ``(updated_ctx, report)``，其中 ``updated_ctx.target``
        反映了累积的 sanitize 结果。
        """
        scene_name = scene or self._default_scene
        if not scene_name:
            raise ValueError("Checker.run requires a scene name (no default_scene set)")

        if rules is None and not overrides:
            compiled = self._compiled.get(scene_name)
            if compiled is None:
                resolved = resolve_scene(scene_name, self._scenes)
                compiled = _compile(resolved)
                self._compiled[scene_name] = compiled
            return self._execute(ctx, compiled)

        # 高级路径：规则替换和/或覆盖 — 即时编译，
        # 不污染每个 scene 的缓存。
        resolved = resolve_scene(scene_name, self._scenes)
        if rules is not None:
            resolved = ResolvedScene(
                name=resolved.name,
                sanitize=resolved.sanitize,
                rules=tuple(_coerce_rule_entry(r) for r in rules),
            )
        if overrides:
            resolved = ResolvedScene(
                name=resolved.name,
                sanitize=_apply_overrides(resolved.sanitize, overrides),
                rules=_apply_overrides(resolved.rules, overrides),
            )
        return self._execute(ctx, _compile(resolved))

    @staticmethod
    def _execute(ctx: CheckContext, scene: _CompiledScene) -> tuple[CheckContext, CheckReport]:
        for spec, fn in scene.sanitize:
            new_target = fn(ctx, spec)
            if new_target != ctx.target:
                ctx = replace(ctx, target=new_target)

        issues: list[Issue] = []
        for spec, fn in scene.rules:
            new_issues = list(fn(ctx, spec))
            issues.extend(new_issues)
            if any(i.severity is Severity.ERROR for i in new_issues):
                break

        return ctx, CheckReport(issues=tuple(issues))

    # ------------------------------------------------------- 属性

    @property
    def source_lang(self) -> str:
        return self._source_lang

    @property
    def target_lang(self) -> str:
        return self._target_lang

    @property
    def default_scene(self) -> str:
        return self._default_scene

    @property
    def scenes(self) -> Mapping[str, SceneConfig]:
        """scene 表的只读视图（访问时不复制）。"""
        return self._scenes_view


# ---------------------------------------------------------------------------
# 工厂 — 为语言对构建 Checker
# ---------------------------------------------------------------------------

_SAME_SCRIPT = dict(short=5.0, medium=3.0, long=2.0, very_long=1.6)
_CJK_TO_LATIN = dict(short=8.0, medium=5.0, long=3.5, very_long=2.5)
_LATIN_TO_CJK = dict(short=4.0, medium=2.5, long=1.8, very_long=1.4)


def _ratio_thresholds(src_family: ScriptFamily, tgt_family: ScriptFamily) -> dict[str, float]:
    """根据跨文字系统方向选择长度比阈值。"""
    if src_family == "cjk" and tgt_family != "cjk":
        return dict(_CJK_TO_LATIN)
    if src_family != "cjk" and tgt_family == "cjk":
        return dict(_LATIN_TO_CJK)
    return dict(_SAME_SCRIPT)


def _build_keyword_pairs(
    src: LangProfile,
    tgt: LangProfile,
) -> list[tuple[list[str], list[str]]]:
    """从概念交集构建跨语言关键词对。"""
    pairs: list[tuple[list[str], list[str]]] = []
    for concept, src_words in src.concept_words.items():
        tgt_words = tgt.concept_words.get(concept)
        if tgt_words:
            pairs.append((list(src_words), list(tgt_words)))
    return pairs


def _build_translate_scene(
    name: str,
    *,
    tgt_lang: str,
    src_profile: LangProfile,
    tgt_profile: LangProfile,
    base: str = "builtin.translate.strict",
) -> SceneConfig:
    """构建一个 scene，通过参数覆盖将各语言配置档案数据
    接入内置翻译预设。"""
    thresholds = _ratio_thresholds(src_profile.script_family, tgt_profile.script_family)
    keyword_pairs = _build_keyword_pairs(src_profile, tgt_profile)

    overrides: dict[str, dict] = {
        "length_ratio": {"params": thresholds},
        "format_artifacts": {
            "params": {
                "hallucination_starts": list(tgt_profile.hallucination_starts),
            }
        },
        "question_mark": {
            "params": {
                "source_marks": list(src_profile.question_marks),
                "expected_marks": list(tgt_profile.question_marks),
            }
        },
        "keywords": {
            "params": {
                "forbidden_terms": list(tgt_profile.forbidden_terms),
                "keyword_pairs": keyword_pairs,
            }
        },
        "cjk_content": {
            "params": {"target_lang": tgt_lang},
        },
    }
    return SceneConfig(name=name, extends=(base,), overrides=overrides)


@lru_cache(maxsize=32)
def default_checker(source_lang: str, target_lang: str, *, profile: str = "strict") -> Checker:
    """为语言对构建默认的 :class:`Checker`。

    返回一个绑定到按语言对 scene ``translate.<src>.<tgt>`` 的 Checker，
    该 scene 继承 ``builtin.translate.<profile>`` 并注入两侧
    :class:`LangProfile` 的覆盖参数。

    ``profile`` 为 ``"strict"`` 或 ``"lenient"``，对应内置预设
    ``builtin.translate.strict`` / ``builtin.translate.lenient``。
    结果按 ``(source_lang, target_lang, profile)`` 缓存。
    """
    base = f"builtin.translate.{profile}"
    src_profile = get_profile(source_lang)
    tgt_profile = get_profile(target_lang)

    scene_name = f"translate.{source_lang}.{target_lang}"
    return Checker(
        source_lang=source_lang,
        target_lang=target_lang,
        scenes={
            scene_name: _build_translate_scene(
                scene_name,
                tgt_lang=target_lang,
                src_profile=src_profile,
                tgt_profile=tgt_profile,
                base=base,
            ),
        },
        default_scene=scene_name,
    )
