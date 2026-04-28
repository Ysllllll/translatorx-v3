# Checker Scene Design

## Goal

Refactor `src/application/checker/` into a lightweight, configurable checking subsystem.

The checker must support:

- Named scenes in YAML, such as `translate.strict`, `translate.lenient`, `subtitle.line`, and `llm.response`.
- Per-call rule selection and rule parameter overrides.
- Use outside translation workflows.
- Existing result types: `Severity`, `Issue`, and `CheckReport`.

It should not become a general validation framework, add dynamic plugin loading, or change translation retry behavior beyond wiring it to the new checker API.

## Design

Use a function-based rule registry instead of one class per rule:

```python
RuleFn = Callable[[CheckContext, RuleSpec], list[Issue]]
```

Core types:

```python
@dataclass(frozen=True)
class CheckContext:
    source: str = ""
    target: str = ""
    source_lang: str = ""
    target_lang: str = ""
    usage: Usage | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RuleSpec:
    name: str
    severity: Severity = Severity.ERROR
    params: Mapping[str, object] = field(default_factory=dict)
```

`CheckContext` is intentionally not translation-specific. Translation checks populate `source`, `target`, language fields, and optional usage data. Other scenes can use only `target` and `metadata`.

Target module layout:

```text
src/application/checker/
  __init__.py
  types.py        # Severity, Issue, CheckReport, CheckContext, RuleSpec
  registry.py     # rule registry and @rule decorator
  rules.py        # built-in rule functions
  config.py       # CheckerConfig, SceneConfig, RuleConfig
  checker.py      # Checker.check_context() and compatibility check()
  factory.py      # thin from_config/default_checker compatibility helpers
  sanitize.py     # text cleanup, kept separate from checker core
  lang/           # existing language profiles, kept for the first pass
```

`factory.py` should stop building multiple rule-object lists. It should only combine config, language profiles, and the built-in registry into a `Checker`.

## Config And API

Example YAML:

```yaml
checker:
  default_scene: translate.strict
  scenes:
    translate.strict:
      rules:
        - empty_target
        - length_bounds
        - length_ratio
        - format_artifacts
        - question_mark
        - keywords
        - output_tokens
        - cjk_content

    translate.lenient:
      extends: translate.strict
      rules:
        length_ratio:
          severity: warning
          params: {short: 8.0, medium: 5.0, long: 3.5, very_long: 2.5}
        question_mark:
          severity: info

    llm.response:
      rules:
        - empty_target
        - markdown_artifacts
        - forbidden_terms
        - trailing_annotation
```

Rule entries support two forms:

- `- length_ratio`: use default severity and params.
- `length_ratio: {severity: warning, params: {...}}`: override defaults.

`extends` copies the parent scene's ordered rules and patches matching rule names. Child-only rules append to the end.

Primary API:

```python
checker.check_context(ctx, scene="translate.strict")
```

Compatibility API:

```python
checker.check(source, translation, usage=usage, scene="translate.strict")
```

Per-call overrides:

```python
checker.check_context(ctx, scene="translate.strict", rules=["empty_target", "length_ratio"])

checker.check_context(
    ctx,
    scene="translate.strict",
    overrides={"length_ratio": {"severity": "warning", "params": {"short": 10.0}}},
)
```

`rules` replaces the selected scene's rule list for that call. `overrides` patches the selected rule specs for that call.

## Runtime Flow

1. `AppConfig` loads `CheckerConfig`.
2. `App.checker(src, tgt)` builds a checker from `AppConfig.checker`.
3. Callers build a `CheckContext`.
4. `Checker` resolves `scene` or `default_scene`.
5. `Checker` applies per-call `rules` and `overrides`.
6. Rule functions run in order and produce a `CheckReport`.

For the first implementation, keep current ERROR short-circuit behavior. Treat `collect_all=True` and full regression scoring as follow-ups unless they are low-risk during implementation.

## Migration

1. Add `CheckContext`, `RuleSpec`, scene config models, and the registry.
2. Port existing rule behavior to registered rule functions.
3. Implement scene inheritance and per-call overrides.
4. Keep `default_checker(source_lang, target_lang)` and `Checker.check(source, translation, ...)` as wrappers.
5. Wire `App.checker()` to honor `AppConfig.checker`.
6. Wire translation to use the configured default scene and configured sanitizer.
7. Keep legacy rule class exports during the migration window where practical.

## Errors And Tests

Fail config validation for unknown scenes, unknown rule names, and invalid severity strings. Rule functions should not hide configuration errors. Optional runtime dependencies, such as Pillow for pixel width, may continue to no-op when unavailable.

Test coverage should include:

- Scene parsing with list and mapping rule forms.
- Scene inheritance and override precedence.
- Per-call `rules` replacement and `overrides` patching.
- Compatibility `checker.check(source, translation, ...)`.
- `App.checker()` honoring `AppConfig.checker`.
- Translation using the configured default scene and sanitizer.
- Behavior parity for the current default translation checks.
