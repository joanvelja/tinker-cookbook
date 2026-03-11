"""Semaphore-gated async OpenAI-compatible LLM client."""

import asyncio
import os
from typing import Literal

import chz
from openai import AsyncOpenAI


@chz.chz
class LLMClientConfig:
    model: str = "gpt-5-mini"
    reasoning_effort: Literal["low", "medium", "high"] | None = "medium"
    max_concurrent: int | None = None
    timeout_s: float = 60.0
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None


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
        extra: dict = {}
        if self._config.reasoning_effort is not None:
            extra["reasoning_effort"] = self._config.reasoning_effort
        else:
            extra["temperature"] = 0
        async with self._semaphore:
            response = await self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **extra,
            )
        content = response.choices[0].message.content
        return content or ""
