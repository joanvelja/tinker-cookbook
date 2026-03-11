"""Tests for tinker_cookbook.scoring module."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from tinker_cookbook.scoring import (
    AmbiguousVerdictError,
    AsyncBinaryJudge,
    BinaryJudgeError,
    BinaryJudgeTemplate,
    JudgeBatch,
)


# ---------------------------------------------------------------------------
# BinaryJudgeTemplate.parse
# ---------------------------------------------------------------------------

class TestBinaryJudgeTemplateParse:
    @pytest.fixture
    def template(self) -> BinaryJudgeTemplate:
        return BinaryJudgeTemplate(
            system="sys",
            user="{question}",
            name="test",
        )

    def test_correct_exact(self, template: BinaryJudgeTemplate):
        assert template.parse("CORRECT") is True

    def test_incorrect_exact(self, template: BinaryJudgeTemplate):
        assert template.parse("INCORRECT") is False

    def test_correct_lowercase(self, template: BinaryJudgeTemplate):
        assert template.parse("correct") is True

    def test_correct_with_trailing_period(self, template: BinaryJudgeTemplate):
        assert template.parse("correct.") is True

    def test_correct_with_trailing_text(self, template: BinaryJudgeTemplate):
        assert template.parse("CORRECT blah blah") is True

    def test_incorrect_with_trailing_text(self, template: BinaryJudgeTemplate):
        assert template.parse("INCORRECT the answer was wrong") is False

    def test_empty_string_raises(self, template: BinaryJudgeTemplate):
        with pytest.raises(AmbiguousVerdictError):
            template.parse("")

    def test_whitespace_only_raises(self, template: BinaryJudgeTemplate):
        with pytest.raises(AmbiguousVerdictError):
            template.parse("   ")

    def test_maybe_raises(self, template: BinaryJudgeTemplate):
        with pytest.raises(AmbiguousVerdictError):
            template.parse("MAYBE")

    def test_mixed_case(self, template: BinaryJudgeTemplate):
        assert template.parse("Correct") is True

    def test_incorrect_mixed_case(self, template: BinaryJudgeTemplate):
        assert template.parse("Incorrect") is False

    def test_multiline_first_token_correct(self, template: BinaryJudgeTemplate):
        assert template.parse("CORRECT\nsome reasoning here") is True

    def test_unicode_gibberish_raises(self, template: BinaryJudgeTemplate):
        with pytest.raises(AmbiguousVerdictError):
            template.parse("\u2603\u2764\u2600")

    def test_leading_whitespace_correct(self, template: BinaryJudgeTemplate):
        assert template.parse("  CORRECT  ") is True

    def test_custom_positive_negative(self):
        t = BinaryJudgeTemplate(
            system="sys", user="{x}", positive="YES", negative="NO"
        )
        assert t.parse("YES") is True
        assert t.parse("NO") is False
        with pytest.raises(AmbiguousVerdictError):
            t.parse("CORRECT")


# ---------------------------------------------------------------------------
# BinaryJudgeTemplate.render
# ---------------------------------------------------------------------------

class TestBinaryJudgeTemplateRender:
    def test_basic_substitution(self):
        t = BinaryJudgeTemplate(
            system="You are a judge.",
            user="Question: {question}\nTarget: {target}\nResponse: {response}",
            name="test",
        )
        system, user = t.render(question="What is 2+2?", target="4", response="4")
        assert system == "You are a judge."
        assert "What is 2+2?" in user
        assert "Target: 4" in user
        assert "Response: 4" in user

    def test_no_placeholders(self):
        t = BinaryJudgeTemplate(system="sys", user="static text")
        system, user = t.render()
        assert system == "sys"
        assert user == "static text"

    def test_missing_key_raises(self):
        t = BinaryJudgeTemplate(system="sys", user="{missing_key}")
        with pytest.raises(KeyError):
            t.render()


# ---------------------------------------------------------------------------
# AsyncBinaryJudge.judge
# ---------------------------------------------------------------------------

class TestAsyncBinaryJudge:
    @pytest.fixture
    def template(self) -> BinaryJudgeTemplate:
        return BinaryJudgeTemplate(
            system="You are a grading assistant.",
            user="Question: {question}\nTarget: {target}\nResponse: {response}",
            name="test",
        )

    @pytest.mark.asyncio
    async def test_judge_correct(self, template: BinaryJudgeTemplate):
        client = AsyncMock()
        client.complete.return_value = "CORRECT"
        judge = AsyncBinaryJudge(client, template)
        result = await judge.judge(question="q", target="a", response="a")
        assert result is True
        client.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_judge_incorrect(self, template: BinaryJudgeTemplate):
        client = AsyncMock()
        client.complete.return_value = "INCORRECT"
        judge = AsyncBinaryJudge(client, template)
        result = await judge.judge(question="q", target="a", response="b")
        assert result is False

    @pytest.mark.asyncio
    async def test_judge_garbage_raises(self, template: BinaryJudgeTemplate):
        client = AsyncMock()
        client.complete.return_value = "I'm not sure about this"
        judge = AsyncBinaryJudge(client, template)
        with pytest.raises(AmbiguousVerdictError):
            await judge.judge(question="q", target="a", response="b")

    @pytest.mark.asyncio
    async def test_judge_passes_rendered_prompt(self, template: BinaryJudgeTemplate):
        client = AsyncMock()
        client.complete.return_value = "CORRECT"
        judge = AsyncBinaryJudge(client, template)
        await judge.judge(question="What is pi?", target="3.14", response="3.14159")
        call_args = client.complete.call_args
        assert call_args[0][0] == "You are a grading assistant."
        assert "What is pi?" in call_args[0][1]
        assert "3.14" in call_args[0][1]


# ---------------------------------------------------------------------------
# JudgeBatch
# ---------------------------------------------------------------------------

class TestJudgeBatch:
    @pytest.fixture
    def mock_judge(self) -> AsyncBinaryJudge:
        client = AsyncMock()
        client.complete.return_value = "CORRECT"
        template = BinaryJudgeTemplate(
            system="sys",
            user="Q: {question} T: {target} R: {response}",
            name="test",
        )
        return AsyncBinaryJudge(client, template)

    @pytest.mark.asyncio
    async def test_add_and_run(self, mock_judge: AsyncBinaryJudge):
        batch = JudgeBatch(mock_judge)
        batch.add("k1", question="q1", target="a1", response="r1")
        batch.add("k2", question="q2", target="a2", response="r2")
        result = await batch.run()
        assert result.values == {"k1": True, "k2": True}
        assert result.errors == {}

    @pytest.mark.asyncio
    async def test_resolve_skips_llm(self, mock_judge: AsyncBinaryJudge):
        batch = JudgeBatch(mock_judge)
        batch.resolve("k1", False)
        batch.add("k2", question="q", target="a", response="r")
        result = await batch.run()
        assert result.values["k1"] is False
        assert result.values["k2"] is True
        # Only one LLM call (k2), not two
        assert mock_judge.client.complete.await_count == 1

    @pytest.mark.asyncio
    async def test_dedup_same_key_same_kwargs(self, mock_judge: AsyncBinaryJudge):
        batch = JudgeBatch(mock_judge)
        batch.add("k1", question="q", target="a", response="r")
        batch.add("k1", question="q", target="a", response="r")  # same key+kwargs
        result = await batch.run()
        assert "k1" in result.values
        assert mock_judge.client.complete.await_count == 1

    def test_dedup_conflicting_kwargs_raises(self, mock_judge: AsyncBinaryJudge):
        batch = JudgeBatch(mock_judge)
        batch.add("k1", question="q", target="a", response="r1")
        with pytest.raises(ValueError, match="different kwargs"):
            batch.add("k1", question="q", target="a", response="r2")

    @pytest.mark.asyncio
    async def test_post_run_add_raises(self, mock_judge: AsyncBinaryJudge):
        batch = JudgeBatch(mock_judge)
        await batch.run()
        with pytest.raises(RuntimeError, match="already executed"):
            batch.add("k1", question="q", target="a", response="r")

    @pytest.mark.asyncio
    async def test_empty_batch_returns_empty(self, mock_judge: AsyncBinaryJudge):
        batch = JudgeBatch(mock_judge)
        result = await batch.run()
        assert result.values == {}
        assert result.errors == {}

    @pytest.mark.asyncio
    async def test_empty_batch_with_resolved_only(self, mock_judge: AsyncBinaryJudge):
        batch = JudgeBatch(mock_judge)
        batch.resolve("k1", True)
        batch.resolve("k2", False)
        result = await batch.run()
        assert result.values == {"k1": True, "k2": False}
        assert result.errors == {}
        assert mock_judge.client.complete.await_count == 0

    @pytest.mark.asyncio
    async def test_partial_failures(self):
        """Some calls succeed, some raise BinaryJudgeError."""
        client = AsyncMock()
        call_count = 0

        async def side_effect(system: str, user: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "CORRECT"
            raise RuntimeError("LLM down")

        client.complete.side_effect = side_effect
        template = BinaryJudgeTemplate(system="sys", user="Q: {question} T: {target} R: {response}", name="test")
        judge = AsyncBinaryJudge(client, template)

        batch = JudgeBatch(judge)
        batch.add("good", question="q1", target="a1", response="r1")
        batch.add("bad", question="q2", target="a2", response="r2")
        result = await batch.run()

        assert result.values["good"] is True
        assert "bad" in result.errors
        assert isinstance(result.errors["bad"], BinaryJudgeError)

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        """CancelledError should be re-raised, not caught."""
        client = AsyncMock()
        client.complete.side_effect = asyncio.CancelledError()
        template = BinaryJudgeTemplate(system="sys", user="{question}", name="test")
        judge = AsyncBinaryJudge(client, template)

        batch = JudgeBatch(judge)
        batch.add("k1", question="q")
        with pytest.raises(asyncio.CancelledError):
            await batch.run()

    @pytest.mark.asyncio
    async def test_double_run_raises(self, mock_judge: AsyncBinaryJudge):
        batch = JudgeBatch(mock_judge)
        await batch.run()
        with pytest.raises(RuntimeError, match="already executed"):
            await batch.run()

    def test_resolve_conflicting_value_raises(self, mock_judge: AsyncBinaryJudge):
        batch = JudgeBatch(mock_judge)
        batch.resolve("k1", True)
        with pytest.raises(ValueError, match="already resolved"):
            batch.resolve("k1", False)

    def test_resolve_same_value_dedupes(self, mock_judge: AsyncBinaryJudge):
        batch = JudgeBatch(mock_judge)
        batch.resolve("k1", True)
        batch.resolve("k1", True)  # no error

    def test_resolve_after_add_raises(self, mock_judge: AsyncBinaryJudge):
        batch = JudgeBatch(mock_judge)
        batch.add("k1", question="q", target="a", response="r")
        with pytest.raises(ValueError, match="already pending"):
            batch.resolve("k1", True)

    def test_add_resolved_key_skipped(self, mock_judge: AsyncBinaryJudge):
        batch = JudgeBatch(mock_judge)
        batch.resolve("k1", True)
        batch.add("k1", question="q", target="a", response="r")  # should be silently skipped
        assert "k1" not in batch._pending

    @pytest.mark.asyncio
    async def test_post_run_resolve_raises(self, mock_judge: AsyncBinaryJudge):
        batch = JudgeBatch(mock_judge)
        await batch.run()
        with pytest.raises(RuntimeError, match="already executed"):
            batch.resolve("k1", True)

    @pytest.mark.asyncio
    async def test_ambiguous_verdict_in_batch(self):
        """AmbiguousVerdictError should appear in errors dict."""
        client = AsyncMock()
        client.complete.return_value = "MAYBE"
        template = BinaryJudgeTemplate(system="sys", user="Q: {question}", name="test")
        judge = AsyncBinaryJudge(client, template)

        batch = JudgeBatch(judge)
        batch.add("k1", question="q")
        result = await batch.run()
        assert "k1" in result.errors
        assert isinstance(result.errors["k1"], BinaryJudgeError)
        assert "k1" not in result.values


# ---------------------------------------------------------------------------
# Adversarial edge-case tests (gatekeeper)
# ---------------------------------------------------------------------------


class TestParseAdversarial:
    """Edge cases targeting parse() robustness."""

    @pytest.fixture
    def template(self) -> BinaryJudgeTemplate:
        return BinaryJudgeTemplate(system="sys", user="{q}", name="adv")

    def test_correct_incorrect_first_token_wins(self, template: BinaryJudgeTemplate):
        """When response starts with CORRECT followed by INCORRECT, first token wins."""
        assert template.parse("CORRECT INCORRECT") is True

    def test_incorrect_correct_first_token_wins(self, template: BinaryJudgeTemplate):
        """When response starts with INCORRECT followed by CORRECT, first token wins."""
        assert template.parse("INCORRECT CORRECT") is False

    def test_tab_separated_tokens(self, template: BinaryJudgeTemplate):
        """Tab-separated first token should still be parsed."""
        assert template.parse("CORRECT\there is my reasoning") is True

    def test_newline_only_raises(self, template: BinaryJudgeTemplate):
        with pytest.raises(AmbiguousVerdictError):
            template.parse("\n\n\n")

    def test_punctuation_only_token(self, template: BinaryJudgeTemplate):
        """A token that is entirely punctuation gets stripped to empty by rstrip."""
        with pytest.raises(AmbiguousVerdictError):
            template.parse("... rest")

    def test_correct_with_heavy_punctuation(self, template: BinaryJudgeTemplate):
        assert template.parse("CORRECT!!!...") is True

    def test_correct_with_trailing_comma(self, template: BinaryJudgeTemplate):
        assert template.parse("CORRECT, the answer is right") is True

    def test_unicode_correct_raises(self, template: BinaryJudgeTemplate):
        """Full-width CORRECT should NOT parse (different codepoints)."""
        with pytest.raises(AmbiguousVerdictError):
            template.parse("\uff23\uff2f\uff32\uff32\uff25\uff23\uff34")  # ＣＯＲＲＥＣＴ

    def test_very_long_response_first_token(self, template: BinaryJudgeTemplate):
        """First token is CORRECT even with 10k chars after."""
        assert template.parse("CORRECT " + "x" * 10_000) is True

    def test_null_byte_in_response(self, template: BinaryJudgeTemplate):
        """Null bytes should not crash parse."""
        with pytest.raises(AmbiguousVerdictError):
            template.parse("\x00CORRECT")

    def test_positive_as_substring_raises(self, template: BinaryJudgeTemplate):
        """CORRECTO is not CORRECT."""
        with pytest.raises(AmbiguousVerdictError):
            template.parse("CORRECTO")

    def test_negative_as_substring_raises(self, template: BinaryJudgeTemplate):
        """INCORRECTLY is not INCORRECT."""
        with pytest.raises(AmbiguousVerdictError):
            template.parse("INCORRECTLY")


class TestJudgeBatchAdversarial:
    """Edge cases targeting JudgeBatch state machine."""

    @pytest.fixture
    def mock_judge(self) -> AsyncBinaryJudge:
        client = AsyncMock()
        client.complete.return_value = "CORRECT"
        template = BinaryJudgeTemplate(
            system="sys", user="Q: {question}", name="test"
        )
        return AsyncBinaryJudge(client, template)

    def test_resolve_then_add_same_key_is_noop(self, mock_judge: AsyncBinaryJudge):
        """resolve(k, v) then add(k, ...) should silently skip -- already tested,
        but verify the resolved value is preserved in run()."""
        batch = JudgeBatch(mock_judge)
        batch.resolve("k1", False)
        batch.add("k1", question="q")
        # k1 should NOT be in pending
        assert "k1" not in batch._pending
        assert batch._resolved["k1"] is False

    @pytest.mark.asyncio
    async def test_many_cancelled_errors(self):
        """Multiple CancelledErrors in gather: first one encountered re-raises."""
        client = AsyncMock()
        client.complete.side_effect = asyncio.CancelledError()
        template = BinaryJudgeTemplate(system="sys", user="{q}", name="t")
        judge = AsyncBinaryJudge(client, template)

        batch = JudgeBatch(judge)
        for i in range(5):
            batch.add(f"k{i}", q=f"q{i}")
        with pytest.raises(asyncio.CancelledError):
            await batch.run()

    @pytest.mark.asyncio
    async def test_batch_result_values_are_copies(self, mock_judge: AsyncBinaryJudge):
        """BatchResult.values should be a separate dict from internal state."""
        batch = JudgeBatch(mock_judge)
        batch.resolve("k1", True)
        batch.add("k2", question="q")
        result = await batch.run()
        # Mutating the result should not affect the batch's internal state
        result.values["k3"] = False
        assert "k3" not in batch._resolved

    @pytest.mark.asyncio
    async def test_all_errors_no_values(self):
        """When every call fails, values should only contain resolved entries."""
        client = AsyncMock()
        client.complete.side_effect = RuntimeError("boom")
        template = BinaryJudgeTemplate(system="sys", user="{q}", name="t")
        judge = AsyncBinaryJudge(client, template)

        batch = JudgeBatch(judge)
        batch.resolve("pre", True)
        batch.add("k1", q="q1")
        batch.add("k2", q="q2")
        result = await batch.run()
        assert result.values == {"pre": True}
        assert len(result.errors) == 2

    def test_hashable_tuple_keys(self, mock_judge: AsyncBinaryJudge):
        """Keys can be any hashable, not just strings."""
        batch = JudgeBatch(mock_judge)
        batch.add((1, "a"), question="q")
        batch.resolve((2, "b"), False)
        assert (1, "a") in batch._pending
        assert (2, "b") in batch._resolved


class TestOpenAIProviderTokenBudget:
    """Test GPT-5 token budget paths without making real API calls."""

    def test_effective_token_budget_gpt5(self):
        from tinker_cookbook.scoring.providers import OpenAIBinaryJudgeClient

        client = OpenAIBinaryJudgeClient(model="gpt-5-mini", max_tokens=8)
        assert client._effective_token_budget("gpt-5-mini", "high") == 16_384

    def test_effective_token_budget_non_gpt5(self):
        from tinker_cookbook.scoring.providers import OpenAIBinaryJudgeClient

        client = OpenAIBinaryJudgeClient(model="gpt-4o", max_tokens=8)
        assert client._effective_token_budget("gpt-4o", "high") == 8

    def test_retry_budgets_gpt5_high_already_large(self):
        """When effective budget >= 16384, no retry candidates are larger."""
        from tinker_cookbook.scoring.providers import OpenAIBinaryJudgeClient

        client = OpenAIBinaryJudgeClient(model="gpt-5", max_tokens=8)
        budgets = client._retry_budgets(16_384, model_name="gpt-5", reasoning_effort="high")
        # 512 and 1024 are both < 16384, so they won't be appended
        assert budgets == [16_384]

    def test_retry_budgets_gpt5_high_small_budget(self):
        """When token_budget is small, retry candidates 512 and 1024 get appended."""
        from tinker_cookbook.scoring.providers import OpenAIBinaryJudgeClient

        client = OpenAIBinaryJudgeClient(model="gpt-5", max_tokens=8)
        budgets = client._retry_budgets(8, model_name="gpt-5", reasoning_effort="high")
        assert budgets == [8, 512, 1024]

    def test_retry_budgets_non_gpt5(self):
        from tinker_cookbook.scoring.providers import OpenAIBinaryJudgeClient

        client = OpenAIBinaryJudgeClient(model="gpt-4o", max_tokens=8)
        budgets = client._retry_budgets(8, model_name="gpt-4o", reasoning_effort="high")
        assert budgets == [8]

    def test_retry_budgets_gpt5_minimal(self):
        """Non-high reasoning effort gets no retries even for gpt-5."""
        from tinker_cookbook.scoring.providers import OpenAIBinaryJudgeClient

        client = OpenAIBinaryJudgeClient(model="gpt-5", max_tokens=8)
        budgets = client._retry_budgets(8, model_name="gpt-5", reasoning_effort="minimal")
        assert budgets == [8]

    def test_should_retry_with_minimal_reasoning(self):
        from tinker_cookbook.scoring.providers import OpenAIBinaryJudgeClient

        client = OpenAIBinaryJudgeClient(model="gpt-5", max_tokens=8)
        assert client._should_retry_with_minimal_reasoning(
            model_name="gpt-5", reasoning_effort="high", output_text=""
        )
        assert not client._should_retry_with_minimal_reasoning(
            model_name="gpt-5", reasoning_effort="high", output_text="CORRECT"
        )
        assert not client._should_retry_with_minimal_reasoning(
            model_name="gpt-5", reasoning_effort="minimal", output_text=""
        )
        assert not client._should_retry_with_minimal_reasoning(
            model_name="gpt-4o", reasoning_effort="high", output_text=""
        )

    def test_response_needs_more_tokens(self):
        from tinker_cookbook.scoring.providers import OpenAIBinaryJudgeClient
        from unittest.mock import MagicMock

        client = OpenAIBinaryJudgeClient(model="gpt-5", max_tokens=8)

        # Incomplete due to max_output_tokens
        resp = MagicMock()
        resp.status = "incomplete"
        resp.incomplete_details = MagicMock()
        resp.incomplete_details.reason = "max_output_tokens"
        assert client._response_needs_more_tokens(resp) is True

        # Complete response
        resp2 = MagicMock()
        resp2.status = "completed"
        resp2.incomplete_details = None
        assert client._response_needs_more_tokens(resp2) is False

        # Incomplete but different reason
        resp3 = MagicMock()
        resp3.status = "incomplete"
        resp3.incomplete_details = {"reason": "content_filter"}
        assert client._response_needs_more_tokens(resp3) is False

    def test_is_responses_fallback_error(self):
        from tinker_cookbook.scoring.providers import OpenAIBinaryJudgeClient

        client = OpenAIBinaryJudgeClient(model="gpt-5", max_tokens=8)
        assert client._is_responses_fallback_error(Exception("404 not found")) is True
        assert client._is_responses_fallback_error(Exception("/responses endpoint error")) is True
        assert client._is_responses_fallback_error(Exception("rate limit exceeded")) is False
        assert client._is_responses_fallback_error(Exception("unknown endpoint")) is True
