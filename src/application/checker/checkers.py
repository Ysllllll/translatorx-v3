"""Checker — scene-driven rule engine for translation quality checking.

The single public entrypoint is :meth:`Checker.run`: it accepts a
:class:`CheckContext`, resolves the scene (built-in or user-supplied),
runs sanitize steps in order to advance ``ctx.target``, then runs check
rules with ERROR-level short-circuit semantics. Returns
``(updated_ctx, report)``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from ._scene import CheckerConfigV2, SceneConfig, resolve_scene
from .registry import build as _build_step
from .types import CheckContext, CheckReport, Issue, ResolvedScene, Severity


class Checker:
    """Scene-driven translation quality checker."""

    __slots__ = (
        "_source_lang",
        "_target_lang",
        "_scenes",
        "_default_scene",
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
        self._scenes = dict(scenes or {})
        self._default_scene = default_scene

    @classmethod
    def from_v2(
        cls,
        config: CheckerConfigV2,
        *,
        source_lang: str = "",
        target_lang: str = "",
    ) -> Checker:
        """Build a Checker bound to a v2 scene config."""
        return cls(
            source_lang=source_lang,
            target_lang=target_lang,
            scenes=config.scenes,
            default_scene=config.default_scene,
        )

    def run(
        self,
        ctx: CheckContext,
        *,
        scene: str | None = None,
        rules: list[Any] | tuple[Any, ...] | None = None,
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

        resolved = resolve_scene(scene_name, self._scenes)

        if rules is not None:
            from ._scene import _coerce_rule_entry

            resolved = ResolvedScene(
                name=resolved.name,
                sanitize=resolved.sanitize,
                rules=tuple(_coerce_rule_entry(r) for r in rules),
            )

        if overrides:
            from ._scene import _apply_overrides

            resolved = ResolvedScene(
                name=resolved.name,
                sanitize=_apply_overrides(resolved.sanitize, overrides),
                rules=_apply_overrides(resolved.rules, overrides),
            )

        return self._execute(ctx, resolved)

    def _execute(self, ctx: CheckContext, scene: ResolvedScene) -> tuple[CheckContext, CheckReport]:
        for spec in scene.sanitize:
            fn = _build_step(spec.name, kind="sanitize", **spec.params)
            new_target = fn(ctx, spec)
            if new_target != ctx.target:
                ctx = replace(ctx, target=new_target)

        issues: list[Issue] = []
        for spec in scene.rules:
            fn = _build_step(spec.name, kind="check", **spec.params)
            new_issues = list(fn(ctx, spec))
            issues.extend(new_issues)
            if any(i.severity is Severity.ERROR for i in new_issues):
                break

        return ctx, CheckReport(issues=tuple(issues))

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
        return dict(self._scenes)
