"""Builder for constructing BinaryJudgeClient instances from config."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import chz
import tinker
from tinker.lib.public_interfaces.service_client import RetryConfig

from tinker_cookbook import model_info
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.scoring.providers import (
    AnthropicBinaryJudgeClient,
    OpenAIBinaryJudgeClient,
    TinkerBinaryJudgeClient,
)
from tinker_cookbook.scoring.types import BinaryJudgeClient
from tinker_cookbook.tokenizer_utils import get_tokenizer

if TYPE_CHECKING:
    from tinker_cookbook.usage import UsageTracker


@chz.chz
class BinaryJudgeBuilder:
    provider: Literal["tinker", "openai_compatible", "anthropic"]
    model: str
    renderer_name: str | None = None
    reasoning_effort: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    max_tokens: int = 8
    temperature: float = 0.0
    max_connections: int = 64
    timeout_s: float = 60.0

    def build(self, *, usage_tracker: UsageTracker | None = None) -> BinaryJudgeClient:
        if self.provider == "tinker":
            renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
                self.model,
                reasoning_effort=self.reasoning_effort,
            )
            renderer = get_renderer(renderer_name, get_tokenizer(self.model))
            service = tinker.ServiceClient(base_url=self.base_url)
            retry_config = RetryConfig(
                max_connections=self.max_connections,
                progress_timeout=self.timeout_s,
            )
            sampling_client = service.create_sampling_client(
                base_model=self.model,
                retry_config=retry_config,
            )
            completer = TinkerMessageCompleter(
                sampling_client=sampling_client,
                renderer=renderer,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                usage_tracker=usage_tracker,
                actor="binary_judge",
                model_name=self.model,
            )
            return TinkerBinaryJudgeClient(completer)

        if self.provider == "openai_compatible":
            return OpenAIBinaryJudgeClient(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
                base_url=self.base_url,
                api_key_env=self.api_key_env,
                timeout_s=self.timeout_s,
            )

        if self.provider == "anthropic":
            return AnthropicBinaryJudgeClient(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                base_url=self.base_url,
                api_key_env=self.api_key_env,
                timeout_s=self.timeout_s,
            )

        raise ValueError(f"Unsupported provider: {self.provider}")
