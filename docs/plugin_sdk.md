# Pipeline Plugin SDK

Third-party packages can contribute custom pipeline stages without
forking translatorx. Stages are picked up at runtime via Python's
[entry-points] mechanism, so a `pip install` is the only "registration"
step a downstream user has to perform.

This document is the contract: stable shapes and stable behaviours.
Anything not described here is an implementation detail.

[entry-points]: https://packaging.python.org/en/latest/specifications/entry-points/

## TL;DR

```toml
# pyproject.toml of your plugin package
[project.entry-points."translatorx.pipeline.stages"]
my_namespace = "my_pkg.stages:register"
```

```python
# my_pkg/stages.py
from typing import Mapping, Any

from pydantic import BaseModel, ConfigDict, Field

from application.pipeline.registry import StageRegistry
from ports.stage import RecordStage


class MyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    threshold: float = Field(0.5, description="Drop records below this score.")


class MyStage(RecordStage):
    def __init__(self, params: MyParams) -> None:
        self._params = params

    async def run(self, ctx, records):
        ...
        return records


def _factory(params: Mapping[str, Any]) -> MyStage:
    return MyStage(MyParams.model_validate(dict(params)))


def register(reg: StageRegistry) -> None:
    reg.register("my_stage", _factory, params_schema=MyParams)
```

The host application activates plugins explicitly:

```python
from application.pipeline.plugins import discover_stages
from application.stages import make_default_registry

reg = make_default_registry(app)
discover_stages(reg)        # walks the entry-points group
```

## Entry-point group

| Group                              | Purpose                          |
| ---------------------------------- | -------------------------------- |
| `translatorx.pipeline.stages`      | All pipeline stages (any layer). |

A future minor release may introduce per-layer groups (`...build`,
`...structure`, `...enrich`) for stricter validation. Until then a
single group is authoritative; this current group **will continue to
work** as a superset.

## Plugin contract

A plugin is a single callable:

```python
def register(reg: StageRegistry) -> None: ...
```

* Called exactly **once** per `StageRegistry` instance.
* Must call `reg.register(name, factory, params_schema=...)` for every
  stage it wants to expose.
* Must not raise on success. Any exception during `register()` is
  treated as a load failure (see `on_error`).
* Should be idempotent at the import level — translatorx may import the
  module before invoking it.

The callable can live anywhere; conventionally it sits in a
`stages.py` (or `__init__.py`) of the plugin package and is referenced
in the entry point as `my_pkg.stages:register`.

## Discovery API

```python
from application.pipeline.plugins import (
    PluginGroup,           # str — default entry-point group name
    PluginLoadError,       # raised on hard failure
    discover_stages,       # main entry-point walker
    load_plugin,           # resolve a single ep → callable
)
```

### `discover_stages(registry, *, group=..., on_error="warn", names=None)`

| Arg         | Meaning                                                     |
| ----------- | ----------------------------------------------------------- |
| `registry`  | Target `StageRegistry`.                                     |
| `group`     | Override the group name (defaults to `PluginGroup`).        |
| `on_error`  | `"warn"` (log + continue), `"raise"` (`PluginLoadError`), `"ignore"`. |
| `names`     | Optional whitelist of entry-point names to load.            |

Returns the names of plugins that loaded successfully.

> Discovery is **opt-in**. `App` does **not** call `discover_stages`
> automatically — embedders decide when (if ever) to enable plugins.
> This keeps tests isolated and prevents surprise imports.

## Stage params and JSON Schema

Every stage exposes a Pydantic v2 `Params` model. The same model is
served by `GET /api/stages/{name}/schema` and embedded into the
pipeline JSON Schema (`GET /api/stages/schema`). To get a clean
front-end UX:

* Set `model_config = ConfigDict(extra="forbid")` so typos surface as
  validation errors.
* Use `Field(..., description="...")` on user-facing parameters.
* Provide `default=` values where sensible; optional parameters become
  optional in the YAML DSL.

## Compatibility promise

* Stage `name` strings are part of your public API — changing one is
  a breaking change for users who have YAML pipelines referencing it.
* `StageRegistry.register` signature is stable across minor releases.
  New keyword arguments may be added; positional ones will not be
  reordered.
* The `PluginGroup` constant will not change without a deprecation
  cycle of at least one minor release.
* `Params` models can add **optional** fields freely. Removing or
  renaming fields is a breaking change.

## Testing your plugin

```python
from application.pipeline.registry import StageRegistry
from my_pkg.stages import register


def test_register_adds_my_stage():
    reg = StageRegistry()
    register(reg)
    assert reg.is_registered("my_stage")
```

For an end-to-end test, build an `App` with your stage in the
registry, drive a tiny pipeline through it, and assert the resulting
records.

## Versioning

Plugins are not coupled to a translatorx wheel version pin — they work
against the public `application.pipeline` API. To avoid breakage in
practice, declare a sensible lower bound in your plugin's
`pyproject.toml`:

```toml
[project]
dependencies = ["translatorx>=0.X,<0.Y"]
```

Bumping the upper bound is part of a normal release cycle.
