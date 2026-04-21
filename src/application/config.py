"""AppConfig — YAML + Pydantic v2 (D-057).

Single source of truth for all app-level settings: engines, contexts,
store layout, runtime budgets, user tiers. ``App.from_config(path)``
parses a YAML file into :class:`AppConfig` and constructs live
engine / store / checker instances.

Environment overrides follow the ``TRX_<SECTION>__<KEY>`` convention
(double underscore between nesting levels). Example::

    TRX_ENGINES__DEFAULT__API_KEY=sk-xxx

Minimal YAML::

    engines:
      default:
        kind: openai_compat
        model: Qwen/Qwen3-32B
        base_url: http://localhost:26592/v1
        api_key_env: QWEN_API_KEY
    store:
      kind: json
      root: ./workspace
    contexts:
      en_zh:
        src: en
        tgt: zh
    runtime:
      default_checker_profile: strict
      max_concurrent_videos: 3
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------


class EngineEntry(BaseModel):
    """One engine definition."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["openai_compat"] = "openai_compat"
    model: str
    base_url: str
    api_key: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout: float = 150.0
    extra_body: dict[str, Any] = Field(default_factory=dict)

    def resolve_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "EMPTY")
        return "EMPTY"


class ContextEntry(BaseModel):
    """One translation context definition (src → tgt pair)."""

    model_config = ConfigDict(extra="forbid")

    src: str
    tgt: str
    window_size: int = 4
    max_retries: int = 3
    system_prompt_template: str = ""
    terms: dict[str, str] = Field(default_factory=dict)


class StoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["json"] = "json"
    root: str = "./workspace"


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_checker_profile: Literal["strict", "lenient", "minimal"] = "strict"
    max_concurrent_videos: int = 3
    flush_every: int = 100


class PreprocessConfig(BaseModel):
    """Preprocessing pipeline configuration (D-073/D-074).

    ``punc_position`` controls where punctuation restoration runs:

    - ``"global"`` — before ``sentences()``; helps sentence splitting accuracy.
    - ``"sentence"`` — after ``sentences()``; fixes per-sentence punctuation,
      may split one sentence into two.
    - ``"both"`` — runs at both positions.

    ``chunk_mode`` controls sentence chunking strategy:

    - ``"spacy"`` — use spaCy NLP model for sentence splitting.
    - ``"llm"`` — recursive binary splitting via LLM.
    - ``"spacy_llm"`` — spaCy coarse split first, then LLM fine split
      for chunks still exceeding ``chunk_len``.
    """

    model_config = ConfigDict(extra="forbid")

    punc_mode: Literal["none", "ner", "llm", "remote"] = "none"
    punc_position: Literal["global", "sentence", "both"] = "global"
    punc_engine: str = "default"
    punc_endpoint: str | None = None
    punc_threshold: int = 180
    punc_max_retries: int = 2
    punc_on_failure: Literal["keep", "raise"] = "keep"

    spacy_model: str = ""

    chunk_mode: Literal["none", "spacy", "llm", "spacy_llm"] = "none"
    chunk_engine: str = "default"
    chunk_len: int = 90
    chunk_max_depth: int = 4
    chunk_max_retries: int = 2
    chunk_on_failure: Literal["rule", "keep", "raise"] = "rule"
    chunk_split_parts: int = 2
    merge_under: int | None = None
    max_len: int | None = None

    max_concurrent: int = 8


class AppConfig(BaseModel):
    """Root configuration model."""

    model_config = ConfigDict(extra="forbid")

    engines: dict[str, EngineEntry] = Field(default_factory=dict)
    contexts: dict[str, ContextEntry] = Field(default_factory=dict)
    store: StoreConfig = Field(default_factory=StoreConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)

    # -- loaders ---------------------------------------------------------

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "AppConfig":
        """Parse a YAML file into an :class:`AppConfig`.

        Environment variables of the form ``TRX_<SECTION>__<KEY>`` override
        the parsed values (uppercase match, case-insensitive section key).
        """
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, text: str) -> "AppConfig":
        """Parse YAML *text* (not a path) into an :class:`AppConfig`."""
        data = yaml.safe_load(text) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """Construct from a plain dict, applying env overrides."""
        data = _apply_env_overrides(dict(data), prefix="TRX_")
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Env override helper
# ---------------------------------------------------------------------------


def _apply_env_overrides(data: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    """Apply ``<prefix><SECTION>__<KEY>`` style env overrides.

    Nested values are reached via ``__`` separators. Leaf values are
    assigned as strings (Pydantic coerces as needed). Missing intermediate
    dicts are created.
    """
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = env_key[len(prefix) :].lower().split("__")
        if not parts:
            continue
        cur: dict[str, Any] = data
        for p in parts[:-1]:
            nxt = cur.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[p] = nxt
            cur = nxt
        cur[parts[-1]] = env_val
    return data


__all__ = [
    "AppConfig",
    "ContextEntry",
    "EngineEntry",
    "PreprocessConfig",
    "RuntimeConfig",
    "StoreConfig",
]
