"""OpenAI-compatible LLM engine.

Works with any API that implements the OpenAI chat completions interface:
OpenAI, DeepSeek, Qwen (vLLM/Ollama), local servers, etc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import AsyncIterator

from openai import AsyncOpenAI

from .._protocol import Message


@dataclass
class EngineConfig:
    """Configuration for an OpenAI-compatible engine.

    Attributes:
        model: Model identifier (e.g. "Qwen/Qwen3-32B").
        base_url: API base URL (e.g. "http://localhost:26592/v1").
        api_key: API key (use "EMPTY" for local servers).
        temperature: Default sampling temperature.
        max_tokens: Default max generation tokens.
        timeout: Request timeout in seconds.
        extra_body: Extra parameters passed to the API (model-specific).
    """

    model: str = ""
    base_url: str = ""
    api_key: str = "EMPTY"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: float = 150.0
    extra_body: dict = field(default_factory=dict)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    # Complete blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Incomplete opening tag (no closing tag — model cut off)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text


def _clean_response(text: str) -> str:
    """Clean common LLM output artifacts."""
    text = _strip_think_tags(text)
    text = re.sub(r"^`+|`+$", "", text)
    text = text.strip()
    return text


class OpenAICompatEngine:
    """Async LLM engine for OpenAI-compatible APIs.

    Satisfies the :class:`~llm_ops.LLMEngine` Protocol.
    """

    __slots__ = ("_client", "_config")

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        base_url = config.base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=base_url,
            timeout=config.timeout,
        )

    @property
    def config(self) -> EngineConfig:
        return self._config

    @property
    def model(self) -> str:
        return self._config.model

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """Send messages and return the full response text."""
        kwargs: dict = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._config.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self._config.extra_body:
            kwargs["extra_body"] = self._config.extra_body
        response = await self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return _clean_response(content)

    async def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield response tokens incrementally."""
        kwargs: dict = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._config.max_tokens,
            "stream": True,
        }
        if self._config.extra_body:
            kwargs["extra_body"] = self._config.extra_body
        response = await self._client.chat.completions.create(**kwargs)
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
