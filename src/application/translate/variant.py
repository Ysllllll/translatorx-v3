"""VariantSpec — identifies one translation flavor.

A *variant* is a named (model, prompt, config) tuple. Persisted
:class:`SentenceRecord` translations are keyed by variant so the same
record can carry several side-by-side translations for A/B comparison::

    record.translations["zh"] = {
        "qwen-baseline": "你好世界",
        "gpt5-strict":   "你好，世界",
    }

The variant *key* (used as the dict key above) is either:

* the user-supplied :attr:`alias` when non-empty, or
* a short SHA-256 prefix derived from ``(model, prompt_id, config)``.

A :class:`TranslateProcessor` writing translations also registers the
variant under ``translate.json/variants[key]`` and the prompt body
under ``translate.json/prompts[prompt_id]`` so the JSON file is
self-describing — no need to consult AppConfig to interpret a key.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


_DEFAULT_PROMPT_ID = "default"


@dataclass(frozen=True)
class VariantSpec:
    """Immutable description of one translation flavor.

    Attributes:
        model: Informational model name (e.g. ``"Qwen/Qwen3-32B"``).
            Stored alongside the variant so consumers can audit which
            model produced which translation. Does not actually drive the
            engine — that's still :class:`LLMEngine`.
        prompt_id: Stable id for the prompt body. When two variants
            share the same ``prompt_id``, they share an entry in the
            top-level ``prompts`` table of ``translate.json``.
        prompt: The system prompt body. Empty means "use the
            :class:`TranslateNodeConfig.system_prompt` fallback or the
            built-in default for the language pair".
        config: Free-form extra config (temperature, terms_version, …).
            Affects :attr:`key` when no :attr:`alias` is given.
        alias: Optional user-supplied display key. When non-empty,
            takes precedence over the auto hash.

    The ``config`` dict is stored verbatim; for hashability, it is
    serialized to a stable JSON string at construction time and that
    string drives :attr:`key`.
    """

    model: str = ""
    prompt_id: str = _DEFAULT_PROMPT_ID
    prompt: str = ""
    config: tuple[tuple[str, Any], ...] = ()
    alias: str = ""

    @classmethod
    def create(
        cls,
        *,
        model: str = "",
        prompt_id: str = _DEFAULT_PROMPT_ID,
        prompt: str = "",
        config: dict[str, Any] | None = None,
        alias: str = "",
    ) -> "VariantSpec":
        """Build a :class:`VariantSpec` from a plain dict-style ``config``.

        ``config`` is sorted and frozen into a tuple so the resulting
        dataclass is hashable / immutable.
        """
        items: tuple[tuple[str, Any], ...] = ()
        if config:
            items = tuple(sorted(config.items()))
        return cls(model=model, prompt_id=prompt_id, prompt=prompt, config=items, alias=alias)

    @property
    def config_dict(self) -> dict[str, Any]:
        """Return :attr:`config` as a plain dict (a fresh copy)."""
        return dict(self.config)

    @property
    def key(self) -> str:
        """Stable variant key.

        ``alias`` if set, else ``sha256(model|prompt_id|config_json)[:8]``.
        ``prompt`` body is intentionally **not** part of the key — two
        variants with same ``prompt_id`` but different prompt text are a
        misconfiguration, not a new variant. (The user picks the prompt
        registry semantics.)
        """
        if self.alias:
            return self.alias
        cfg_repr = json.dumps(dict(self.config), sort_keys=True, ensure_ascii=False)
        sig = f"model={self.model}|pid={self.prompt_id}|cfg={cfg_repr}"
        return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:8]

    def info(self) -> dict[str, Any]:
        """Serialize key_info for the ``variants`` registry in ``translate.json``."""
        out: dict[str, Any] = {}
        if self.model:
            out["model"] = self.model
        if self.prompt_id:
            out["prompt_id"] = self.prompt_id
        if self.config:
            out["config"] = dict(self.config)
        return out


_DEFAULT_VARIANT: VariantSpec = VariantSpec()


def default_variant() -> VariantSpec:
    """Return a sentinel default :class:`VariantSpec`.

    Useful when callers don't care about variants but the API still
    requires one (e.g. tests that just want translations to round-trip).
    """
    return _DEFAULT_VARIANT


__all__ = ["VariantSpec", "default_variant"]
