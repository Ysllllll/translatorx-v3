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
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from application.scheduler import TenantQuota

from ports.backpressure import ChannelConfig, OverflowPolicy

from .checker.config import CheckerConfig


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

    kind: Literal["json", "sqlite"] = "json"
    root: str = "./workspace"
    sqlite_path: str | None = None
    """Optional SQLite DB path (only used when ``kind='sqlite'``).

    If unset, defaults to ``<root>/translatorx.sqlite``.
    """


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_checker_profile: Literal["strict", "lenient", "minimal"] = "strict"
    max_concurrent_videos: int = 3
    flush_every: int = 100
    flush_interval_s: float = float("inf")


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
    - ``"spacy_llm_rule"`` — three-stage chain: spaCy → LLM → rule
      length-splitter as a hard backstop for any surviving oversize
      chunks.
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

    chunk_mode: Literal["none", "spacy", "llm", "spacy_llm", "spacy_llm_rule"] = "none"
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
    tenant: str | None = None


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


class ChannelConfigEntry(BaseModel):
    """Pydantic mirror of :class:`ports.backpressure.ChannelConfig`.

    Pydantic models give us YAML/env validation; the entry materializes
    into the frozen ``ChannelConfig`` dataclass via :meth:`build`.
    """

    model_config = ConfigDict(extra="forbid")

    capacity: int = Field(default=64, ge=1)
    high_watermark: float = Field(default=0.8, ge=0.0, le=1.0)
    low_watermark: float = Field(default=0.3, ge=0.0, le=1.0)
    overflow: Literal["block", "drop_new", "drop_old", "reject"] = "block"

    def build(self) -> ChannelConfig:
        return ChannelConfig(
            capacity=self.capacity,
            high_watermark=self.high_watermark,
            low_watermark=self.low_watermark,
            overflow=OverflowPolicy(self.overflow),
        )


class BusConfigEntry(BaseModel):
    """Pydantic mirror of :class:`ports.message_bus.BusConfig`.

    Only ``redis_streams`` requires a non-empty ``url``. ``memory``
    means: do not build a remote bus — :class:`PipelineRuntime` falls
    back to in-process :class:`MemoryChannel` for every stage pair.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["memory", "redis_streams"] = "memory"
    url: str | None = None
    consumer_group: str = "trx-runners"
    consumer_name: str | None = None
    block_ms: int = Field(default=5000, ge=0)
    max_in_flight: int = Field(default=64, ge=1)

    def build(self) -> "BusConfig":
        from ports.message_bus import BusConfig

        return BusConfig(
            type=self.type,
            url=self.url,
            consumer_group=self.consumer_group,
            consumer_name=self.consumer_name,
            block_ms=self.block_ms,
            max_in_flight=self.max_in_flight,
        )


class StreamingConfig(BaseModel):
    """Streaming-runtime defaults (Phase 3).

    ``default_channel`` is applied to every stage pair in
    :meth:`PipelineRuntime.stream` unless a stage carries its own
    ``downstream_channel`` override in the YAML DSL.

    ``bus`` (Phase 4 J5) is optional. When set with ``type=redis_streams``,
    stages flagged with ``bus_topic`` route through a cross-process
    :class:`BusChannel` instead of in-process :class:`MemoryChannel`.
    Default ``type=memory`` keeps the runtime fully local.
    """

    model_config = ConfigDict(extra="forbid")

    default_channel: ChannelConfigEntry = Field(default_factory=ChannelConfigEntry)
    bus: BusConfigEntry = Field(default_factory=BusConfigEntry)


class HotReloadConfig(BaseModel):
    """Hot-reload watcher for :attr:`AppConfig.pipelines_dir`.

    Default is OFF. When ``enabled=True``, the App spawns a watcher that
    invalidates :meth:`App.pipelines` cache whenever a YAML file in the
    directory changes (added/modified/removed).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    backend: Literal["poll", "watchdog"] = "poll"
    interval_s: float = Field(default=2.0, gt=0.0)


class TenantQuotaEntry(BaseModel):
    """Pydantic mirror of :class:`application.scheduler.TenantQuota`.

    Phase 5 (方案 L). Configurable per tenant under :attr:`AppConfig.tenants`;
    materialized into the frozen ``TenantQuota`` dataclass via :meth:`build`.
    """

    model_config = ConfigDict(extra="forbid")

    max_concurrent_streams: int = Field(default=1, ge=1)
    max_qps: float = Field(default=1.0, gt=0.0)
    qos_tier: Literal["free", "standard", "premium"] = "free"
    cost_budget_usd_per_min: float | None = Field(default=None, ge=0.0)

    def build(self) -> "TenantQuota":
        """Materialize into the frozen :class:`TenantQuota` dataclass."""
        from application.scheduler import TenantQuota

        return TenantQuota(
            max_concurrent_streams=self.max_concurrent_streams,
            max_qps=self.max_qps,
            qos_tier=self.qos_tier,
            cost_budget_usd_per_min=self.cost_budget_usd_per_min,
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
    checker: CheckerConfig = Field(default_factory=CheckerConfig)

    pipelines_dir: str | None = None
    """Optional directory of named pipeline YAML files (one pipeline per
    file). Discovered files are loaded into :attr:`pipelines` lazily by
    :meth:`App.pipelines`. Phase 2 (D) MVP."""

    pipelines: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Inline named pipelines as raw dicts. Each value is the same shape
    accepted by :func:`application.pipeline.loader.load_pipeline_dict`.
    Materialized into :class:`PipelineDef` on demand by callers."""

    hot_reload: "HotReloadConfig" = Field(default_factory=lambda: HotReloadConfig())
    """Optional file-system watcher for ``pipelines_dir``. Default OFF for
    production safety. Phase 2 (D) — see :mod:`application.pipeline.hot_reload`."""

    streaming: "StreamingConfig" = Field(default_factory=lambda: StreamingConfig())
    """Streaming-runtime defaults (Phase 3). The ``default_channel`` is
    applied to every adjacent stage pair in
    :meth:`PipelineRuntime.stream` unless a stage carries its own
    ``downstream_channel`` override in the YAML DSL."""

    tenants: dict[str, TenantQuotaEntry] = Field(default_factory=dict)
    """Per-tenant streaming quotas (Phase 5 — 方案 L). Keyed by
    ``tenant_id``. Tenants not listed here fall back to
    :data:`application.scheduler.DEFAULT_QUOTAS["free"]`. Materialized by
    :meth:`build_tenant_quotas` into ``dict[str, TenantQuota]`` for the
    scheduler."""

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

    def build_tenant_quotas(self) -> "dict[str, TenantQuota]":
        """Materialize :attr:`tenants` into the frozen scheduler dataclass map.

        Returns a fresh ``dict[str, TenantQuota]`` keyed by tenant id.
        Callers (e.g. :class:`api.app.App` / :class:`FairScheduler`) merge
        this with :data:`application.scheduler.DEFAULT_QUOTAS` so unlisted
        tenants still receive a sane default.
        """
        return {tid: entry.build() for tid, entry in self.tenants.items()}


# ---------------------------------------------------------------------------
# Env override helper
# ---------------------------------------------------------------------------


def _apply_env_overrides(data: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    """Apply ``<prefix><SECTION>__<KEY>`` style env overrides.

    Nested values are reached via ``__`` separators. Leaf values are
    assigned as strings (Pydantic coerces as needed). Missing intermediate
    dicts are created.

    C20 — when an intermediate path resolves to a non-dict scalar in the
    source data, raising is preferred over silently overwriting it: a
    typo like ``TRX_STORE__ROOT__SUB=...`` against ``store.root: str``
    would otherwise turn ``root`` into a fresh dict and lose the user's
    configured value.
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
            if nxt is None:
                nxt = {}
                cur[p] = nxt
            elif not isinstance(nxt, dict):
                raise ValueError(
                    f"env override {env_key!r} expects a nested mapping at "
                    f"{'.'.join(parts[: parts.index(p) + 1])!r} but the config "
                    f"provides a scalar of type {type(nxt).__name__}. "
                    "Refusing to silently overwrite."
                )
            cur = nxt
        # Final-leaf scalar/dict mismatch: refuse to overwrite a dict
        # with a string value (would clobber a whole sub-tree).
        existing = cur.get(parts[-1])
        if isinstance(existing, dict):
            raise ValueError(
                f"env override {env_key!r} would replace the dict at "
                f"{'.'.join(parts)!r} with a scalar string. Use the dotted "
                "leaf form instead (e.g. set the individual sub-keys)."
            )
        cur[parts[-1]] = env_val
    return data


__all__ = [
    "AppConfig",
    "AuthKeyEntry",
    "ChannelConfigEntry",
    "ContextEntry",
    "EngineEntry",
    "HotReloadConfig",
    "PreprocessConfig",
    "RuntimeConfig",
    "ServiceConfig",
    "StoreConfig",
    "StreamingConfig",
    "TranscriberConfig",
    "TTSConfig",
]
