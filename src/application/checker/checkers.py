"""Checker — scene-driven rule engine for translation quality checking.

Two public entrypoints:

- :meth:`Checker.check` — high-level convenience; takes raw ``source`` /
  ``target`` strings and returns ``(sanitized_target, report)``. Callers
  never construct :class:`CheckContext` themselves.
- :meth:`Checker.run`   — low-level; takes a fully-built
  :class:`CheckContext` (needed when the caller wants to pass ``usage`` /
  ``prior`` / ``metadata``). Returns ``(updated_ctx, report)``.

Rule callables are compiled **once per scene** (keyed by scene name) and
cached on the Checker instance, so per-call cost is just running the
compiled functions — no factory invocation, no regex recompilation, no
font reload.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

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
    """Resolved scene with rule callables already instantiated."""

    name: str
    sanitize: tuple[tuple[RuleSpec, Callable[..., str]], ...]
    rules: tuple[tuple[RuleSpec, Callable[..., Any]], ...]


def _compile(resolved: ResolvedScene) -> _CompiledScene:
    sanitize = tuple((spec, _build_step(spec.name, kind="sanitize", **spec.params)) for spec in resolved.sanitize)
    rules = tuple((spec, _build_step(spec.name, kind="check", **spec.params)) for spec in resolved.rules)
    return _CompiledScene(name=resolved.name, sanitize=sanitize, rules=rules)


_EMPTY_METADATA: Mapping[str, Any] = MappingProxyType({})


class Checker:
    """Scene-driven translation quality checker."""

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
        # Cache: scene name → compiled callables. Only populated for the
        # "no overrides / no rules-replace" hot path. Advanced run() calls
        # bypass the cache and compile on the fly.
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
        """Build a Checker bound to a :class:`CheckerConfig`."""
        return cls(
            source_lang=source_lang,
            target_lang=target_lang,
            scenes=config.scenes,
            default_scene=config.default_scene,
        )

    # Backwards-compatible alias (the ``v2`` suffix was a migration tag).
    from_v2 = from_config

    # ---------------------------------------------------------------- run

    def check(
        self,
        source: str,
        target: str,
        *,
        usage: Any = None,
        prior: str | None = None,
        scene: str | None = None,
        source_lang: str | None = None,
        target_lang: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[str, CheckReport]:
        """High-level entrypoint — operate on raw strings.

        Internally builds a :class:`CheckContext` (using the Checker's
        bound ``source_lang`` / ``target_lang`` unless explicitly
        overridden), runs sanitize + check, and returns
        ``(sanitized_target, report)``. Callers that don't need
        ``usage`` / ``prior`` / ``metadata`` should prefer this.
        """
        ctx = CheckContext(
            source=source,
            target=target,
            source_lang=self._source_lang if source_lang is None else source_lang,
            target_lang=self._target_lang if target_lang is None else target_lang,
            usage=usage,
            prior=prior,
            metadata=metadata or _EMPTY_METADATA,
        )
        new_ctx, report = self.run(ctx, scene=scene)
        return new_ctx.target, report

    def run(
        self,
        ctx: CheckContext,
        *,
        scene: str | None = None,
        rules: Sequence[Any] | None = None,
        overrides: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[CheckContext, CheckReport]:
        """Run sanitize + check pipeline for *ctx* under *scene*.

        Resolution order:

        1. ``scene`` argument > ``self._default_scene``. If neither is
           set, raises :class:`ValueError`.
        2. ``rules=[...]`` *replaces* the resolved rule list (sanitize
           steps from the scene still run).
        3. ``overrides={name: {severity?, params?}}`` apply on top.

        Returns ``(updated_ctx, report)`` where ``updated_ctx.target``
        reflects the cumulative sanitize result.
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

        # Advanced path: rules-replace and/or overrides — compile fresh,
        # don't pollute the per-scene cache.
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

    # ------------------------------------------------------- properties

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
        """Read-only view of the scene table (no copy on access)."""
        return self._scenes_view
