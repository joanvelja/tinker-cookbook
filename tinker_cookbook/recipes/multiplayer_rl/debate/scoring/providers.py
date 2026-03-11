"""Provider adapters for binary semantic judging in debate scoring."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import chz
import tinker
from tinker.lib.public_interfaces.service_client import RetryConfig

from tinker_cookbook import model_info
from tinker_cookbook.completers import MessageCompleter, TinkerMessageCompleter
from tinker_cookbook.renderers import format_content_as_string, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

if TYPE_CHECKING:
    from tinker_cookbook.usage import UsageTracker


class AnswerJudgeClient(Protocol):
    async def complete_binary(
        self,
        *,
        system: str,
        user: str,
        kind: str | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class BinaryJudgeCallRecord:
    kind: str | None
    system: str
    user: str
    response: str
    timestamp_utc: str


@dataclass
class RecordingAnswerJudgeClient:
    inner: AnswerJudgeClient
    calls: list[BinaryJudgeCallRecord] = field(default_factory=list)

    async def complete_binary(
        self,
        *,
        system: str,
        user: str,
        kind: str | None = None,
    ) -> str:
        response = await self.inner.complete_binary(system=system, user=user, kind=kind)
        self.calls.append(
            BinaryJudgeCallRecord(
                kind=kind,
                system=system,
                user=user,
                response=response,
                timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        )
        return response

    def write_jsonl(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(
                {
                    "timestamp_utc": record.timestamp_utc,
                    "kind": record.kind,
                    "system": record.system,
                    "user": record.user,
                    "response": record.response,
                }
            )
            + "\n"
            for record in self.calls
        ]
        output_path.write_text("".join(lines))


@dataclass
class TinkerAnswerJudgeClient:
    completer: MessageCompleter

    async def complete_binary(
        self,
        *,
        system: str,
        user: str,
        kind: str | None = None,
    ) -> str:
        response = await self.completer(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        return format_content_as_string(response["content"], separator="")


@dataclass
class OpenAICompatibleAnswerJudgeClient:
    model: str
    max_tokens: int = 8
    temperature: float = 0.0
    reasoning_effort: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_s: float = 60.0
    _client: object | None = field(init=False, default=None, repr=False)
    _GPT5_SCORER_OUTPUT_BUDGET: int = field(init=False, default=16_384, repr=False)

    def _retry_budgets(self, token_budget: int, *, model_name: str, reasoning_effort: str | None) -> list[int]:
        budgets = [token_budget]
        if model_name.startswith("gpt-5") and reasoning_effort == "high":
            for candidate in (512, 1024):
                if candidate > budgets[-1]:
                    budgets.append(candidate)
        return budgets

    def _is_responses_fallback_error(self, exc: Exception) -> bool:
        error_text = str(exc).lower()
        return any(
            needle in error_text
            for needle in (
                "/responses",
                "responses.create",
                "unsupported parameter",
                "not found",
                "404",
                "unknown url",
                "unknown endpoint",
                "unsupported endpoint",
            )
        )

    def _response_needs_more_tokens(self, response: object) -> bool:
        status = getattr(response, "status", None)
        incomplete_details = getattr(response, "incomplete_details", None)
        if isinstance(incomplete_details, dict):
            reason = incomplete_details.get("reason")
        else:
            reason = getattr(incomplete_details, "reason", None)
        return status == "incomplete" and reason == "max_output_tokens"

    def _should_retry_with_minimal_reasoning(
        self,
        *,
        model_name: str,
        reasoning_effort: str | None,
        output_text: str,
    ) -> bool:
        return model_name.startswith("gpt-5") and reasoning_effort == "high" and output_text == ""

    def _effective_token_budget(self, model_name: str, reasoning_effort: str | None) -> int:
        token_budget = self.max_tokens
        if model_name.startswith("gpt-5"):
            # GPT-5 can spend the visible output budget on reasoning for long
            # matcher/grader payloads. Use a large floor so binary verdicts
            # remain observable even on verbose answer comparisons.
            token_budget = max(token_budget, self._GPT5_SCORER_OUTPUT_BUDGET)
        return token_budget

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError(
                "openai is required for provider='openai_compatible'. "
                "Install tinker_cookbook[debate-scorers]."
            ) from exc

        api_key = os.environ.get(self.api_key_env) if self.api_key_env else None
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout_s,
        )
        return self._client

    async def complete_binary(
        self,
        *,
        system: str,
        user: str,
        kind: str | None = None,
    ) -> str:
        client = self._get_client()
        model_name = self.model.lower()
        reasoning_effort = self.reasoning_effort
        if reasoning_effort is None and model_name.startswith("gpt-5"):
            reasoning_effort = "minimal"
        response_token_budget = self._effective_token_budget(model_name, reasoning_effort)
        chat_token_budget = self.max_tokens
        if hasattr(client, "responses"):
            try:
                response_text = await self._complete_via_responses(
                    client,
                    system=system,
                    user=user,
                    model_name=model_name,
                    reasoning_effort=reasoning_effort,
                    token_budget=response_token_budget,
                )
                if self._should_retry_with_minimal_reasoning(
                    model_name=model_name,
                    reasoning_effort=reasoning_effort,
                    output_text=response_text,
                ):
                    minimal_reasoning = "minimal"
                    return await self._complete_via_responses(
                        client,
                        system=system,
                        user=user,
                        model_name=model_name,
                        reasoning_effort=minimal_reasoning,
                        token_budget=self._effective_token_budget(model_name, minimal_reasoning),
                    )
                return response_text
            except Exception as exc:
                # Fall back to chat completions for OpenAI-compatible providers that
                # do not support /responses.
                if not self._is_responses_fallback_error(exc):
                    raise

        request_kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.temperature not in (0, 0.0):
            request_kwargs["temperature"] = self.temperature
        if reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = reasoning_effort

        response = await self._create_completion(
            client,
            request_kwargs=request_kwargs,
            token_budget=chat_token_budget,
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content

    async def _complete_via_responses(
        self,
        client,
        *,
        system: str,
        user: str,
        model_name: str,
        reasoning_effort: str | None,
        token_budget: int,
    ) -> str:
        for budget in self._retry_budgets(
            token_budget,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
        ):
            response_kwargs = {
                "model": self.model,
                "instructions": system,
                "input": user,
                "max_output_tokens": budget,
            }
            if self.temperature not in (0, 0.0):
                response_kwargs["temperature"] = self.temperature
            if reasoning_effort is not None:
                response_kwargs["reasoning"] = {"effort": reasoning_effort}

            response = await client.responses.create(**response_kwargs)
            output_text = getattr(response, "output_text", None)
            if isinstance(output_text, str) and output_text:
                return output_text
            if not self._response_needs_more_tokens(response):
                return ""
        return ""

    async def _create_completion(
        self,
        client,
        *,
        request_kwargs: dict[str, object],
        token_budget: int,
    ):
        try:
            return await client.chat.completions.create(
                **request_kwargs,
                max_completion_tokens=token_budget,
            )
        except Exception as exc:
            error_text = str(exc)
            if "reasoning_effort" in error_text and "reasoning_effort" in request_kwargs:
                without_reasoning = dict(request_kwargs)
                without_reasoning.pop("reasoning_effort", None)
                return await self._create_completion(
                    client,
                    request_kwargs=without_reasoning,
                    token_budget=token_budget,
                )
            if "max_completion_tokens" not in error_text:
                raise
        return await client.chat.completions.create(
            **request_kwargs,
            max_tokens=token_budget,
        )


@dataclass
class AnthropicAnswerJudgeClient:
    model: str
    max_tokens: int = 8
    temperature: float = 0.0
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_s: float = 60.0
    _client: object | None = field(init=False, default=None, repr=False)

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError(
                "anthropic is required for provider='anthropic'. "
                "Install tinker_cookbook[debate-scorers]."
            ) from exc

        api_key = os.environ.get(self.api_key_env) if self.api_key_env else None
        kwargs = {"api_key": api_key, "timeout": self.timeout_s}
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        self._client = AsyncAnthropic(**kwargs)
        return self._client

    async def complete_binary(
        self,
        *,
        system: str,
        user: str,
        kind: str | None = None,
    ) -> str:
        client = self._get_client()
        response = await client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        chunks = []
        for block in response.content:
            block_text = getattr(block, "text", None)
            if block_text:
                chunks.append(block_text)
        return "".join(chunks)


@chz.chz
class DebateScorerBuilder:
    provider: Literal["tinker", "openai_compatible", "anthropic"]
    model: str
    renderer_name: str | None = None
    reasoning_effort: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    max_tokens: int = 8
    temperature: float = 0.0
    max_connections: int = 64
    timeout_s: int = 60

    def build(self, *, usage_tracker: UsageTracker | None = None) -> AnswerJudgeClient:
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
                actor="semantic_scorer",
                model_name=self.model,
            )
            return TinkerAnswerJudgeClient(completer)

        if self.provider == "openai_compatible":
            return OpenAICompatibleAnswerJudgeClient(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
                base_url=self.base_url,
                api_key_env=self.api_key_env,
                timeout_s=float(self.timeout_s),
            )

        if self.provider == "anthropic":
            return AnthropicAnswerJudgeClient(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                base_url=self.base_url,
                api_key_env=self.api_key_env,
                timeout_s=float(self.timeout_s),
            )

        raise ValueError(f"Unsupported scorer provider: {self.provider}")
