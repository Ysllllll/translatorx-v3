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


class AlignConfig(BaseModel):
    """AlignProcessor configuration.

    Governs the binary-recursive alignment pass that maps an N-segment
    source window onto the LLM translation. Two modes:

    - JSON mode — LLM returns structured ``{mapping: [{src, tgt}, ...]}``.
    - Text mode — two-line plain text output; enables source word-level
      ``rearrange`` via :func:`domain.subtitle.rebalance_segment_words`.

    ``norm_ratio`` is the cross-length tolerance for a clean accept.
    Ratios in ``[norm_ratio, accept_ratio)`` are accepted but flagged
    with ``need_rearrange`` so the text pass can rebalance word boundaries.
    """

    model_config = ConfigDict(extra="forbid")

    engine: str = "default"
    enable_text_mode: bool = False
    json_norm_ratio: float = 5.0
    json_accept_ratio: float = 5.0
    text_norm_ratio: float = 3.0
    text_accept_ratio: float = 3.0
    rearrange_chunk_len: int = 90
    max_concurrent: int = 8


class TranscriberConfig(BaseModel):
    """Transcriber backend configuration (Stage 6).

    Two flavors are bundled:

    - ``library="whisperx"`` — local WhisperX model (GPU).
    - ``library="openai"`` — OpenAI-compatible ``/v1/audio/transcriptions``.
    - ``library="http"`` — self-hosted HTTP WhisperX-style service.

    Leave ``library`` empty to disable transcription entirely.
    """

    model_config = ConfigDict(extra="allow")

    library: Literal["", "whisperx", "openai", "http"] = ""
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    language: str | None = None
    word_timestamps: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)


class TTSConfig(BaseModel):
    """TTS backend configuration (Stage 6).

    The backend is resolved via :func:`adapters.tts.create` using the
    ``library`` field. Arbitrary extra keys pass through to the
    backend-specific factory.
    """

    model_config = ConfigDict(extra="allow")

    library: Literal["", "edge-tts", "openai-tts", "elevenlabs", "local"] = ""
    default_voice: str = ""
    format: str = "mp3"
    rate: float = 1.0
    api_key: str = ""
    base_url: str = ""
    speaker_map: dict[str, str] = Field(default_factory=dict)
    gender_map: dict[str, Literal["male", "female", "neutral"]] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


class AuthKeyEntry(BaseModel):
    """One API key → principal mapping under :attr:`ServiceConfig.api_keys`."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    tier: str = "free"


class ServiceConfig(BaseModel):
    """FastAPI service configuration (Stage 7).

    Consumed by :func:`api.service.create_app` to wire auth + the
    resource manager. Leave all fields empty/default for a dev server
    (no auth, in-memory resource manager).
    """

    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = 8080
    api_keys: dict[str, AuthKeyEntry] = Field(
        default_factory=dict,
        description="Mapping of API key string → principal (user_id + tier).",
    )
    resource_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = ""
    redis_key_prefix: str = "trx:rm:"

    task_backend: Literal["inproc", "arq"] = "inproc"
    arq_queue_name: str = "trx:tasks"
    arq_task_prefix: str = "trx:task:"
    arq_events_prefix: str = "trx:task-events:"

    prometheus_enabled: bool = False
    prometheus_path: str = "/metrics"
    otel_enabled: bool = False
    otel_service_name: str = "translatorx"
    otel_exporter: Literal["console", "otlp-grpc", "otlp-http"] = "console"
    otel_endpoint: str = ""

    cors_origins: list[str] = Field(
        default_factory=list,
        description="Allowed CORS origins. Empty = CORS disabled. Use ['*'] to allow all.",
    )
    cors_allow_credentials: bool = False

    task_persist_path: str = Field(
        default="",
        description=(
            "Directory for persistent task metadata (JSON per task). "
            "Empty = no persistence (in-proc only). Relative paths resolve "
            "under the store root."
        ),
    )

    rps_limit: float = Field(
        default=0.0,
        description="Per-user requests/second. <=0 disables the limiter.",
    )
    rps_burst: int = Field(
        default=0,
        description="Per-user bucket capacity. <=0 falls back to rps_limit.",
    )

    request_log_enabled: bool = True
    reload_enabled: bool = False
    reload_config_path: str = ""
    error_log_path: str = Field(
        default="",
        description="JSONL error log path. Empty = in-memory buffer only.",
    )


class AppConfig(BaseModel):
    """Root configuration model."""

    model_config = ConfigDict(extra="forbid")

    engines: dict[str, EngineEntry] = Field(default_factory=dict)
    contexts: dict[str, ContextEntry] = Field(default_factory=dict)
    store: StoreConfig = Field(default_factory=StoreConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    align: AlignConfig = Field(default_factory=AlignConfig)
    transcriber: TranscriberConfig = Field(default_factory=TranscriberConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)

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
    "AuthKeyEntry",
    "ContextEntry",
    "EngineEntry",
    "PreprocessConfig",
    "RuntimeConfig",
    "ServiceConfig",
    "StoreConfig",
    "TranscriberConfig",
    "TTSConfig",
]
