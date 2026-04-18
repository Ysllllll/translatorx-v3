"""Tests for llm_ops.engines.openai_compat — OpenAICompatEngine."""

from __future__ import annotations

import pytest

from llm_ops.protocol import LLMEngine
from llm_ops.engines.openai_compat import (
    EngineConfig,
    OpenAICompatEngine,
    _clean_response,
    _strip_think_tags,
)


# ---------------------------------------------------------------------------
# Unit tests — response cleaning
# ---------------------------------------------------------------------------

class TestStripThinkTags:
    def test_removes_think_block(self):
        assert _strip_think_tags("<think>内部思考</think>翻译结果") == "翻译结果"

    def test_removes_incomplete_open_tag(self):
        assert _strip_think_tags("<think>thinking but no close") == ""

    def test_removes_incomplete_with_prefix(self):
        assert _strip_think_tags("result<think>trailing thought") == "result"

    def test_no_think_tags(self):
        assert _strip_think_tags("normal text") == "normal text"

    def test_empty(self):
        assert _strip_think_tags("") == ""

    def test_multiple_think_blocks(self):
        text = "<think>a</think>middle<think>b</think>end"
        result = _strip_think_tags(text)
        assert "middle" in result
        assert "end" in result
        assert "<think>" not in result


class TestCleanResponse:
    def test_strips_backticks(self):
        assert _clean_response("```翻译结果```") == "翻译结果"

    def test_strips_leading_backticks(self):
        assert _clean_response("```\n翻译结果") == "翻译结果"

    def test_strips_whitespace(self):
        assert _clean_response("  hello  \n") == "hello"

    def test_combined_cleanup(self):
        raw = "<think>let me think</think>```翻译结果```\n"
        assert _clean_response(raw) == "翻译结果"


# ---------------------------------------------------------------------------
# EngineConfig
# ---------------------------------------------------------------------------

class TestEngineConfig:
    def test_defaults(self):
        cfg = EngineConfig()
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 2048
        assert cfg.api_key == "EMPTY"
        assert cfg.timeout == 150.0

    def test_custom(self):
        cfg = EngineConfig(
            model="Qwen/Qwen3-32B",
            base_url="http://localhost:26592",
            temperature=0.5,
            max_tokens=4096,
        )
        assert cfg.model == "Qwen/Qwen3-32B"
        assert cfg.temperature == 0.5

    def test_extra_body(self):
        cfg = EngineConfig(
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        assert "chat_template_kwargs" in cfg.extra_body


# ---------------------------------------------------------------------------
# OpenAICompatEngine — construction & protocol
# ---------------------------------------------------------------------------

class TestOpenAICompatEngine:
    def test_satisfies_protocol(self):
        cfg = EngineConfig(model="test", base_url="http://localhost:8000")
        engine = OpenAICompatEngine(cfg)
        assert isinstance(engine, LLMEngine)

    def test_base_url_normalization(self):
        cfg = EngineConfig(model="test", base_url="http://localhost:8000/")
        engine = OpenAICompatEngine(cfg)
        assert engine._client.base_url.host == "localhost"

    def test_base_url_with_v1(self):
        cfg = EngineConfig(model="test", base_url="http://localhost:8000/v1")
        engine = OpenAICompatEngine(cfg)
        assert "v1" in str(engine._client.base_url)

    def test_model_property(self):
        cfg = EngineConfig(model="Qwen/Qwen3-32B", base_url="http://localhost:8000")
        engine = OpenAICompatEngine(cfg)
        assert engine.model == "Qwen/Qwen3-32B"

    def test_config_property(self):
        cfg = EngineConfig(model="test", base_url="http://localhost:8000")
        engine = OpenAICompatEngine(cfg)
        assert engine.config is cfg


# ---------------------------------------------------------------------------
# Integration test — requires running LLM server
# ---------------------------------------------------------------------------

@pytest.mark.skipif(True, reason="Requires live LLM server; run manually")
class TestOpenAICompatEngineLive:
    """Enable by setting the skip condition to False and ensuring a server runs."""

    @pytest.mark.asyncio
    async def test_complete(self):
        cfg = EngineConfig(
            model="Qwen/Qwen3-32B",
            base_url="http://localhost:26592",
            temperature=0.3,
            max_tokens=100,
        )
        engine = OpenAICompatEngine(cfg)
        result = await engine.complete([{"role": "user", "content": "Say hello in Chinese"}])
        assert len(result.text) > 0
        assert result.usage is None or result.usage.model == "Qwen/Qwen3-32B"

    @pytest.mark.asyncio
    async def test_stream(self):
        cfg = EngineConfig(
            model="Qwen/Qwen3-32B",
            base_url="http://localhost:26592",
            temperature=0.3,
            max_tokens=100,
        )
        engine = OpenAICompatEngine(cfg)
        chunks = []
        async for chunk in engine.stream([{"role": "user", "content": "Say hello"}]):
            chunks.append(chunk)
        assert len(chunks) > 0
