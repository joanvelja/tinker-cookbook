"""
Implementations that correspond to a model or policy that can be sampled from, but with different amounts of additional structure.

The TokenCompleter operates on tokens. This is the version used by RL algorithms, because RL algorithms work on Tokens. The MessageCompleter operates on messages, so it needs to be used with a renderer.

Evals and other code should use the appropriate interface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

import tinker

from tinker_cookbook import renderers

if TYPE_CHECKING:
    from tinker_cookbook.usage import UsageTracker

# Interfaces

StopCondition: TypeAlias = list[str] | list[int]


@dataclass
class TokensWithLogprobs:
    tokens: list[int]
    maybe_logprobs: list[float] | None

    @property
    def logprobs(self) -> list[float]:
        if self.maybe_logprobs is None:
            raise ValueError("Logprobs are not available")
        return self.maybe_logprobs


class TokenCompleter:
    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        raise NotImplementedError


class MessageCompleter:
    # TODO maybe add n_samples to the interfaces?
    async def __call__(self, messages: list[renderers.Message]) -> renderers.Message:
        raise NotImplementedError


# Implementations


@dataclass
class TinkerTokenCompleter(TokenCompleter):
    """
    The most standard TokenCompleter, which uses a tinker.SamplingClient to sample actions.
    """

    sampling_client: tinker.SamplingClient
    max_tokens: int
    temperature: float = 1.0
    usage_tracker: UsageTracker | None = field(default=None, repr=False)
    actor: str = "trained"
    model_name: str = ""

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        """Sample an action from the policy given an observation."""
        # Sample from the model
        t0 = time.monotonic()
        sample_result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )
        self._last_sample_wall_s = time.monotonic() - t0

        # Extract tokens and logprobs from the first (and only) sample
        sampled_tokens = sample_result.sequences[0].tokens
        sampled_logprobs = sample_result.sequences[0].logprobs
        assert sampled_logprobs is not None

        self._last_input_tokens = model_input.length
        self._last_output_tokens = len(sampled_tokens)

        if self.usage_tracker is not None:
            from tinker_cookbook.usage import UsageEvent

            self.usage_tracker.record(
                UsageEvent(
                    actor=self.actor,
                    model_name=self.model_name,
                    input_tokens=model_input.length,
                    output_tokens=len(sampled_tokens),
                )
            )

        return TokensWithLogprobs(tokens=sampled_tokens, maybe_logprobs=sampled_logprobs)


class TinkerMessageCompleter(MessageCompleter):
    """A completer that uses the actual model to generate responses."""

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: renderers.Renderer,
        max_tokens: int,
        stop_condition: StopCondition | None = None,
        temperature: float = 1.0,
        usage_tracker: UsageTracker | None = None,
        actor: str = "",
        model_name: str = "",
    ):
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.usage_tracker = usage_tracker
        self.actor = actor
        self.model_name = model_name
        if stop_condition is None:
            self.stop_condition = self.renderer.get_stop_sequences()
        else:
            self.stop_condition = stop_condition

    async def __call__(self, messages: list[renderers.Message]) -> renderers.Message:
        # Render the conversation for the model
        model_input = self.renderer.build_generation_prompt(messages)

        # Sample from the model
        t0 = time.monotonic()
        response = await self.sampling_client.sample_async(
            model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=self.stop_condition,
            ),
        )
        self._last_sample_wall_s = time.monotonic() - t0

        output_tokens = response.sequences[0].tokens

        self._last_input_tokens = model_input.length
        self._last_output_tokens = len(output_tokens)

        if self.usage_tracker is not None:
            from tinker_cookbook.usage import UsageEvent

            self.usage_tracker.record(
                UsageEvent(
                    actor=self.actor,
                    model_name=self.model_name,
                    input_tokens=model_input.length,
                    output_tokens=len(output_tokens),
                )
            )

        # Decode the response
        parsed_message, _success = self.renderer.parse_response(output_tokens)

        result: renderers.Message = {"role": "assistant", "content": parsed_message["content"]}
        if "tool_calls" in parsed_message:
            result["tool_calls"] = parsed_message["tool_calls"]
        return result
