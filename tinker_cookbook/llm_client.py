"""Semaphore-gated async OpenAI-compatible LLM client."""

import asyncio
import os
from typing import Any, Literal

import chz
from openai import AsyncOpenAI


@chz.chz
class LLMClientConfig:
    model: str = "gpt-5-mini"
    reasoning_effort: Literal["low", "medium", "high", "none"] | None = "medium"
    max_concurrent: int | None = None
    timeout_s: float = 60.0
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    extra_body: dict[str, Any] | None = None


class AsyncLLMClient:
    def __init__(self, config: LLMClientConfig) -> None:
        assert config.max_concurrent is not None, "max_concurrent must be set"
        self._config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._client = AsyncOpenAI(
            api_key=os.environ[config.api_key_env],
            base_url=config.base_url,
            max_retries=2,
            timeout=config.timeout_s,
        )

    async def complete(self, *, system: str, user: str) -> str:
        cfg = self._config
        extra: dict[str, Any] = {}

        # Reasoning effort: "none" is sent literally (OpenRouter uses it to
        # disable thinking); None means "don't send, use temperature instead".
        if cfg.reasoning_effort is not None:
            extra["reasoning_effort"] = cfg.reasoning_effort

        # Temperature: explicit value wins, else 0 when no reasoning_effort.
        if cfg.temperature is not None:
            extra["temperature"] = cfg.temperature
        elif cfg.reasoning_effort is None:
            extra["temperature"] = 0

        if cfg.top_p is not None:
            extra["top_p"] = cfg.top_p
        if cfg.presence_penalty is not None:
            extra["presence_penalty"] = cfg.presence_penalty
        if cfg.max_tokens is not None:
            extra["max_tokens"] = cfg.max_tokens
        if cfg.stop is not None:
            extra["stop"] = cfg.stop
        if cfg.top_k is not None:
            # top_k is not in the OpenAI API; passed via extra_body for providers that support it
            extra.setdefault("extra_body", {})["top_k"] = cfg.top_k

        # Merge extra_body (provider pinning, etc.)
        if cfg.extra_body is not None:
            extra.setdefault("extra_body", {}).update(cfg.extra_body)

        async with self._semaphore:
            response = await self._client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **extra,
            )
        content = response.choices[0].message.content
        return content or ""
