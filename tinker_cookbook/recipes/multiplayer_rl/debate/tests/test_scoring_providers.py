from __future__ import annotations

import asyncio

import httpx
from openai import BadRequestError

from tinker_cookbook.recipes.multiplayer_rl.debate.scoring.providers import (
    OpenAICompatibleAnswerJudgeClient,
)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if "max_completion_tokens" in kwargs:
            raise RuntimeError("Unsupported parameter: 'max_completion_tokens'")
        return _FakeResponse("SAME")


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeChatCompletions()


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = _FakeFailingResponses()
        self.chat = _FakeChat()


class _FakeResponsesResponse:
    def __init__(
        self,
        output_text: str,
        *,
        status: str = "completed",
        incomplete_reason: str | None = None,
    ) -> None:
        self.output_text = output_text
        self.status = status
        self.incomplete_details = (
            None if incomplete_reason is None else {"reason": incomplete_reason}
        )


def _make_bad_request_error(message: str) -> BadRequestError:
    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError(message, response=response, body={"message": message})


class _FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponsesResponse("SAME")


class _FakeRetryingResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return _FakeResponsesResponse(
                "",
                status="incomplete",
                incomplete_reason="max_output_tokens",
            )
        return _FakeResponsesResponse("SAME")


class _FakeAlwaysIncompleteResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponsesResponse(
            "",
            status="incomplete",
            incomplete_reason="max_output_tokens",
        )


class _FakeLongPromptAlwaysIncompleteResponses:
    def __init__(self, *, min_total_prompt_chars: int) -> None:
        self.calls: list[dict[str, object]] = []
        self.min_total_prompt_chars = min_total_prompt_chars

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        total_prompt_chars = len(str(kwargs["instructions"])) + len(str(kwargs["input"]))
        if total_prompt_chars < self.min_total_prompt_chars:
            raise AssertionError("Expected a long prompt input in this regression test")
        return _FakeResponsesResponse(
            "",
            status="incomplete",
            incomplete_reason="max_output_tokens",
        )


class _FakeHighThenMinimalResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        reasoning = kwargs.get("reasoning") or {}
        effort = reasoning.get("effort")
        if effort == "high":
            return _FakeResponsesResponse(
                "",
                status="incomplete",
                incomplete_reason="max_output_tokens",
            )
        if effort == "minimal":
            return _FakeResponsesResponse("SAME")
        raise AssertionError(f"Unexpected reasoning effort: {effort!r}")


class _FakeFailingResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        raise RuntimeError("/responses unsupported")


class _FakeOpenAIResponsesClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()
        self.chat = _FakeChat()


class _FakeOpenAIRetryingResponsesClient:
    def __init__(self) -> None:
        self.responses = _FakeRetryingResponses()
        self.chat = _FakeChat()


class _AlwaysFailingChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        raise RuntimeError(
            "Could not finish the message because max_tokens or model output limit was reached. "
            "Please try again with higher max_tokens."
        )


class _AlwaysFailingChat:
    def __init__(self) -> None:
        self.completions = _AlwaysFailingChatCompletions()


class _FakeOpenAIIncompleteResponsesClient:
    def __init__(self) -> None:
        self.responses = _FakeAlwaysIncompleteResponses()
        self.chat = _AlwaysFailingChat()


class _FakeOpenAILongPromptIncompleteResponsesClient:
    def __init__(self, *, min_total_prompt_chars: int) -> None:
        self.responses = _FakeLongPromptAlwaysIncompleteResponses(
            min_total_prompt_chars=min_total_prompt_chars
        )
        self.chat = _FakeChat()


class _FakeOpenAIHighThenMinimalResponsesClient:
    def __init__(self) -> None:
        self.responses = _FakeHighThenMinimalResponses()
        self.chat = _AlwaysFailingChat()


class _FakeChatCompletionsRejectingRaisedBudget:
    def __init__(self, accepted_max_tokens: int) -> None:
        self.accepted_max_tokens = accepted_max_tokens
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if "max_completion_tokens" in kwargs:
            raise RuntimeError("Unsupported parameter: 'max_completion_tokens'")
        if kwargs["max_tokens"] > self.accepted_max_tokens:
            raise _make_bad_request_error("max_tokens/model output limit was reached")
        return _FakeResponse("SAME")


class _FakeChatRejectingRaisedBudget:
    def __init__(self, accepted_max_tokens: int) -> None:
        self.completions = _FakeChatCompletionsRejectingRaisedBudget(accepted_max_tokens)


class _FakeOpenAIFallbackClient:
    def __init__(self, accepted_max_tokens: int) -> None:
        self.responses = _FakeFailingResponses()
        self.chat = _FakeChatRejectingRaisedBudget(accepted_max_tokens)


def test_openai_compatible_client_falls_back_to_max_tokens():
    client = OpenAICompatibleAnswerJudgeClient(model="gpt-test")
    fake = _FakeOpenAIClient()
    client._client = fake

    result = asyncio.run(client.complete_binary(system="sys", user="usr"))

    assert result == "SAME"
    assert len(fake.chat.completions.calls) == 2
    assert "max_completion_tokens" in fake.chat.completions.calls[0]
    assert "max_tokens" in fake.chat.completions.calls[1]


def test_openai_compatible_client_bumps_gpt5_high_reasoning_budget():
    client = OpenAICompatibleAnswerJudgeClient(
        model="gpt-5-mini",
        max_tokens=32,
        reasoning_effort="high",
    )
    fake = _FakeOpenAIResponsesClient()
    client._client = fake

    result = asyncio.run(client.complete_binary(system="sys", user="usr"))

    assert result == "SAME"
    assert len(fake.responses.calls) == 1
    assert fake.responses.calls[0]["max_output_tokens"] == 16_384


def test_openai_compatible_client_retries_responses_before_chat_fallback():
    client = OpenAICompatibleAnswerJudgeClient(
        model="gpt-5-mini",
        max_tokens=32,
        reasoning_effort="high",
    )
    fake = _FakeOpenAIRetryingResponsesClient()
    client._client = fake

    result = asyncio.run(client.complete_binary(system="sys", user="usr"))

    assert result == "SAME"
    assert [call["max_output_tokens"] for call in fake.responses.calls] == [16_384, 16_384]
    assert [call.get("reasoning", {}).get("effort") for call in fake.responses.calls] == [
        "high",
        "minimal",
    ]
    assert fake.chat.completions.calls == []


def test_openai_compatible_client_does_not_fall_back_to_chat_for_empty_responses():
    client = OpenAICompatibleAnswerJudgeClient(
        model="gpt-5-mini",
        max_tokens=32,
        reasoning_effort="high",
    )
    fake = _FakeOpenAIIncompleteResponsesClient()
    client._client = fake

    result = asyncio.run(client.complete_binary(system="sys", user="usr"))

    assert result == ""
    assert [call["max_output_tokens"] for call in fake.responses.calls] == [16_384, 16_384]
    assert [call.get("reasoning", {}).get("effort") for call in fake.responses.calls] == [
        "high",
        "minimal",
    ]
    assert fake.chat.completions.calls == []


def test_openai_compatible_client_retries_minimal_reasoning_after_empty_high_response():
    client = OpenAICompatibleAnswerJudgeClient(
        model="gpt-5-mini",
        max_tokens=32,
        reasoning_effort="high",
    )
    fake = _FakeOpenAIHighThenMinimalResponsesClient()
    client._client = fake

    result = asyncio.run(client.complete_binary(system="sys", user="usr"))

    assert result == "SAME"
    assert [call.get("reasoning", {}).get("effort") for call in fake.responses.calls] == [
        "high",
        "minimal",
    ]
    assert [call["max_output_tokens"] for call in fake.responses.calls] == [16_384, 16_384]
    assert fake.chat.completions.calls == []


def test_openai_compatible_client_returns_empty_after_long_prompt_responses_budget_exhaustion():
    client = OpenAICompatibleAnswerJudgeClient(
        model="gpt-5-mini",
        max_tokens=32,
        reasoning_effort="high",
    )
    long_system = "semantic scorer system prompt " * 300
    long_user = "semantic scorer user prompt " * 600
    fake = _FakeOpenAILongPromptIncompleteResponsesClient(
        min_total_prompt_chars=len(long_system) + len(long_user)
    )
    client._client = fake

    result = asyncio.run(client.complete_binary(system=long_system, user=long_user))

    assert result == ""
    assert [call["max_output_tokens"] for call in fake.responses.calls] == [16_384, 16_384]
    assert [call.get("reasoning", {}).get("effort") for call in fake.responses.calls] == [
        "high",
        "minimal",
    ]
    assert fake.chat.completions.calls == []


def test_openai_compatible_client_chat_fallback_recovers_from_output_limit_bad_request():
    client = OpenAICompatibleAnswerJudgeClient(
        model="gpt-5-mini",
        max_tokens=32,
        reasoning_effort="high",
    )
    fake = _FakeOpenAIFallbackClient(accepted_max_tokens=32)
    client._client = fake

    result = asyncio.run(client.complete_binary(system="sys", user="usr"))

    assert result == "SAME"
    assert len(fake.responses.calls) == 1
    assert "max_completion_tokens" in fake.chat.completions.calls[0]
    assert fake.chat.completions.calls[-1]["max_tokens"] == 32
