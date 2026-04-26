"""VariantSpec.key resolution tests."""

from __future__ import annotations

import re

from application.translate import VariantSpec


class TestVariantKey:
    def test_alias_wins(self) -> None:
        v = VariantSpec.create(model="Qwen/Qwen3-32B", alias="qwen-baseline")
        assert v.key == "qwen-baseline"

    def test_model_only_uses_readable_form(self) -> None:
        # Just a model name → the model name itself is the key.
        v = VariantSpec.create(model="Qwen/Qwen3-32B")
        assert v.key == "Qwen/Qwen3-32B"

    def test_model_plus_default_prompt_id_omits_default(self) -> None:
        v = VariantSpec.create(model="gpt-5", prompt_id="default")
        assert v.key == "gpt-5"

    def test_model_plus_custom_prompt_id_joins_with_colon(self) -> None:
        v = VariantSpec.create(model="gpt-5", prompt_id="strict")
        assert v.key == "gpt-5:strict"

    def test_config_appends_short_hash(self) -> None:
        v = VariantSpec.create(model="gpt-5", config={"temperature": 0.7})
        # "gpt-5@" + 6-hex
        assert v.key.startswith("gpt-5@")
        assert re.fullmatch(r"gpt-5@[0-9a-f]{6}", v.key)

    def test_config_changes_hash(self) -> None:
        a = VariantSpec.create(model="gpt-5", config={"temperature": 0.7})
        b = VariantSpec.create(model="gpt-5", config={"temperature": 0.3})
        assert a.key != b.key

    def test_long_readable_falls_back_to_full_hash(self) -> None:
        long_model = "x" * 80
        v = VariantSpec.create(model=long_model)
        # Readable form would be 80 chars > 64 → falls back to 8-hex hash.
        assert re.fullmatch(r"[0-9a-f]{8}", v.key)

    def test_empty_spec_falls_back_to_hash(self) -> None:
        v = VariantSpec()
        # No model, no prompt_id override, no alias → fall back.
        assert re.fullmatch(r"[0-9a-f]{8}", v.key)
