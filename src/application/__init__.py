"""Application Layer (L3) — orchestration & use-case implementations.

This layer sits between :mod:`adapters` / :mod:`ports` (below) and
:mod:`api` (above). It owns *use-case logic* and *runtime
orchestration* but performs no direct I/O — adapters do that.

Architecture (three concentric tiers)
=====================================

::

  +-------------------------------------------------------------+
  |  Tier 4 — RUNTIME / DSL                                     |
  |    pipeline/   PipelineRuntime · StageRegistry              |
  |    stages/     <Name>Stage adapters (wrap processors)       |
  +-------------------------------------------------------------+
  |  Tier 3 — STREAMING TRANSFORMERS                            |
  |    processors/ <Name>Processor (ProcessorBase[In, Out])     |
  +-------------------------------------------------------------+
  |  Tier 2 — DOMAIN USE CASES                                  |
  |    align/ summary/ terminology/ translate/                  |
  +-------------------------------------------------------------+

For every domain feature there is a 1:1:1 mapping::

    TranslateProcessor ↔ TranslateStage ↔ "translate" in StageRegistry
    AlignProcessor     ↔ AlignStage     ↔ "align"     in StageRegistry
    SummaryProcessor   ↔ SummaryStage   ↔ "summary"   in StageRegistry
    TTSProcessor       ↔ TTSStage       ↔ "tts"       in StageRegistry

* **processors/** — stateless stream transformers; receive
  :class:`~application.session.VideoSession` via :class:`PipelineContext`.
* **stages/** — pydantic-validated, late-binding adapters for the
  pipeline DSL (one per processor). Lazy-construct the underlying
  processor when the pipeline starts.
* **pipeline/** — DSL runtime: parses YAML, builds stages, manages
  inter-stage channels, emits :class:`DomainEvent` lifecycle events.

Notes on conventions
--------------------

* ``translate`` is exposed as a function (``translate_with_verify``)
  rather than an ``Agent`` class — single LLM call with retry, no
  state to carry.
* ``tts`` has **no** use-case agent; it is pure I/O wrapping
  :class:`ports.tts.TTS`. ``processors/tts.py`` calls the TTS adapter
  directly.

State management
================

* :class:`application.session.VideoSession` — Unit-of-Work aggregate
  for one video's persistent state (load, hydrate, patch, flush).
  Shared across all processors in a single run; supplied by builders
  via :class:`PipelineContext`. Processors never instantiate it.

Quality validation
==================

* :mod:`application.checker` — scene-based rules engine for
  post-translate QA. Consumed by ``processors/translate`` and
  ``stages/registry`` only — Align / Summary / TTS do not run it.

Eventing
========

Two orthogonal axes — **data plane** vs **event plane**:

* *Data plane* (records flowing between stages):
  :class:`MemoryChannel` (single process) and :class:`BusChannel`
  (distributed, redis-backed) — both implement
  :class:`ports.backpressure.BoundedChannel`.
* *Event plane* (lifecycle / observability notifications):
  :class:`DomainEvent` published through :class:`EventBus`
  (in :mod:`application.events`).

Resource governance
===================

* :mod:`application.resources` — per-user / per-tenant budgets and
  quotas (consumed by :mod:`api.app`).
* :mod:`application.scheduler` — fair queueing & per-tenant
  concurrency limits for :class:`PipelineRuntime` (Phase 5).

Layering rules
==============

The ``application`` package may import from ``ports``,
``adapters``, ``domain``, but **never** from ``api``. Cross-package
imports inside ``application`` follow this internal hierarchy
(higher tier may import lower tier; not the reverse)::

    pipeline / stages
        ↓
    processors
        ↓
    align / summary / translate / terminology / checker / session
        ↓
    events / resources / scheduler  (leaf — no application deps)
"""
