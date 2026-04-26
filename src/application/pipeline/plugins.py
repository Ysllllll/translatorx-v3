"""Entry-points based plugin discovery for pipeline stages.

Third-party packages register custom stages via ``setup.cfg`` /
``pyproject.toml`` entry points::

    [project.entry-points."translatorx.pipeline.stages"]
    my_stage = "my_pkg.stages:make_my_stage"

Each entry point must resolve to a callable with signature::

    register(reg: StageRegistry) -> None

The callable is invoked exactly once per registry; it should call
``reg.register(name, factory, params_schema=...)`` for every stage
the plugin contributes.

Discovery is **opt-in**: callers must explicitly invoke
:func:`discover_stages` (typically right after constructing a
registry). This keeps test isolation simple and avoids surprise
imports during ``import translatorx``.
"""

from __future__ import annotations

import logging
from importlib import metadata as importlib_metadata
from typing import Callable, Iterable

from .registry import StageRegistry

__all__ = [
    "PluginGroup",
    "PluginLoadError",
    "discover_stages",
    "load_plugin",
]

logger = logging.getLogger(__name__)

PluginGroup = "translatorx.pipeline.stages"
"""Default entry-point group. Phase-1 single tier; Phase 2 may split
build/structure/enrich into separate groups for stricter validation."""


PluginCallable = Callable[[StageRegistry], None]


class PluginLoadError(RuntimeError):
    """Raised when an entry point cannot be loaded or fails to register."""


def discover_stages(
    registry: StageRegistry,
    *,
    group: str = PluginGroup,
    on_error: str = "warn",
    names: Iterable[str] | None = None,
) -> list[str]:
    """Walk entry points in ``group`` and call each plugin's register fn.

    Parameters
    ----------
    registry :
        Target :class:`StageRegistry`. Each plugin's ``register`` callable
        receives this instance and should add its stages.
    group :
        Entry-point group name. Defaults to :data:`PluginGroup`.
    on_error :
        ``"warn"`` (default) — log and continue on per-plugin failures;
        ``"raise"`` — re-raise as :class:`PluginLoadError` immediately;
        ``"ignore"`` — silently skip failing plugins.
    names :
        Optional whitelist; if set, only entry points whose ``name``
        appears in this iterable are loaded.

    Returns the names of plugins that loaded successfully.
    """
    if on_error not in ("warn", "raise", "ignore"):
        raise ValueError(f"on_error must be 'warn'/'raise'/'ignore', got {on_error!r}")

    eps = importlib_metadata.entry_points(group=group)
    wanted = set(names) if names is not None else None

    loaded: list[str] = []
    for ep in eps:
        if wanted is not None and ep.name not in wanted:
            continue
        try:
            plugin = load_plugin(ep)
            plugin(registry)
        except Exception as exc:  # noqa: BLE001 — boundary
            msg = f"Failed to load pipeline plugin {ep.name!r} ({ep.value}): {exc}"
            if on_error == "raise":
                raise PluginLoadError(msg) from exc
            if on_error == "warn":
                logger.warning(msg)
            continue
        loaded.append(ep.name)

    return loaded


def load_plugin(ep) -> PluginCallable:  # type: ignore[no-untyped-def]
    """Resolve an entry point to its register callable.

    Validates that the resolved object is callable. Type checking on
    the signature is deferred to call site (Python's duck typing).
    """
    obj = ep.load()
    if not callable(obj):
        raise PluginLoadError(
            f"entry point {ep.name!r} resolved to {type(obj).__name__}, expected a callable",
        )
    return obj  # type: ignore[return-value]
