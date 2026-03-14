"""Tests for do_batched_group_rollout — the num_samples=G optimization."""

import asyncio
from dataclasses import dataclass
from functools import partial
from typing import Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest
import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.rollouts import do_batched_group_rollout, do_group_rollout
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup
from tinker_cookbook.renderers.base import Message


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _MockTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


class MockRenderer:
    """Minimal renderer: each char = one token (ord value)."""

    def __init__(self) -> None:
        self.tokenizer = _MockTokenizer()

    def get_stop_sequences(self) -> list[str]:
        return ["<stop>"]

    def build_generation_prompt(
        self, messages: list[Message], prefill: str | None = None
    ) -> tinker.ModelInput:
        text = "".join(m.get("content", "") or "" for m in messages)
        if prefill:
            text += prefill
        tokens = [ord(c) for c in text] if text else []
        return tinker.ModelInput.from_ints(tokens)

    def parse_response(self, tokens: list[int]) -> tuple[Message, bool]:
        text = "".join(chr(t) for t in tokens)
        return Message(role="assistant", content=text), True


class SimpleProblemEnv(ProblemEnv):
    """Deterministic ProblemEnv for testing. Rewards based on token sum parity."""

    def __init__(self, question: str, answer: str, renderer: MockRenderer):
        super().__init__(renderer=renderer)
        self.question = question
        self.answer = answer

    def get_question(self) -> str:
        return self.question

    def get_reference_answer(self) -> str:
        return self.answer

    async def check_answer(self, sample_str: str) -> bool:
        return sample_str.strip() == self.answer

    def check_format(self, sample_str: str) -> bool:
        return len(sample_str) > 0


@dataclass
class FakeSampledSequence:
    tokens: list[int]
    logprobs: list[float]
    stop_reason: str = "stop"


@dataclass
class FakeSampleResponse:
    sequences: list[FakeSampledSequence]
    prompt_logprobs: list[float] | None = None
    topk_prompt_logprobs: list[list[tuple[int, float]]] | None = None


def _make_fake_sampling_client(
    responses_per_sample: list[list[tuple[list[int], list[float]]]],
) -> tinker.SamplingClient:
    """Create a mock SamplingClient.

    responses_per_sample: each element is a list of (tokens, logprobs) tuples
    representing the sequences returned for one sample_async call.
    """
    call_idx = 0

    async def fake_sample_async(
        prompt: tinker.ModelInput,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
        **kwargs,
    ) -> FakeSampleResponse:
        nonlocal call_idx
        idx = min(call_idx, len(responses_per_sample) - 1)
        response_data = responses_per_sample[idx]
        assert len(response_data) == num_samples, (
            f"Expected {num_samples} sequences, got {len(response_data)}"
        )
        call_idx += 1
        return FakeSampleResponse(
            sequences=[
                FakeSampledSequence(tokens=tokens, logprobs=logprobs)
                for tokens, logprobs in response_data
            ]
        )

    client = MagicMock(spec=tinker.SamplingClient)
    client.sample_async = fake_sample_async
    return client


def _tokens(text: str) -> list[int]:
    return [ord(c) for c in text]


def _logprobs(n: int) -> list[float]:
    return [-0.5] * n


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


class TestBatchedGroupRollout:
    """Core tests for do_batched_group_rollout."""

    def test_basic_batched_rollout(self):
        """Batched rollout produces G trajectories with correct structure."""
        renderer = MockRenderer()
        G = 4
        answer_texts = ["yes", "no", "yes", "maybe"]
        responses = [
            (_tokens(text), _logprobs(len(text))) for text in answer_texts
        ]

        client = _make_fake_sampling_client([responses])
        builder = ProblemGroupBuilder(
            env_thunk=partial(
                SimpleProblemEnv,
                question="What?",
                answer="yes",
                renderer=renderer,
            ),
            num_envs=G,
        )

        sampling_params = tinker.SamplingParams(
            stop=["<stop>"], max_tokens=100, temperature=1.0
        )
        result = _run(do_batched_group_rollout(builder, client, sampling_params))

        assert isinstance(result, TrajectoryGroup)
        assert len(result.trajectories_G) == G
        assert len(result.final_rewards_G) == G
        assert len(result.metrics_G) == G

        # Each trajectory should have exactly 1 transition (single-turn)
        for traj in result.trajectories_G:
            assert len(traj.transitions) == 1
            assert traj.transitions[0].episode_done is True

    def test_rewards_match_grading(self):
        """Rewards from batched rollout match what the env would produce."""
        renderer = MockRenderer()
        G = 3
        # "yes" is the correct answer
        answer_texts = ["yes", "no", "yes"]
        responses = [
            (_tokens(text), _logprobs(len(text))) for text in answer_texts
        ]

        client = _make_fake_sampling_client([responses])
        builder = ProblemGroupBuilder(
            env_thunk=partial(
                SimpleProblemEnv,
                question="What?",
                answer="yes",
                renderer=renderer,
            ),
            num_envs=G,
        )

        sampling_params = tinker.SamplingParams(
            stop=["<stop>"], max_tokens=100, temperature=1.0
        )
        result = _run(do_batched_group_rollout(builder, client, sampling_params))

        total_rewards = result.get_total_rewards()
        # "yes" -> correct (check_answer=True, check_format=True) -> reward = 0.1*(1-1) + 1.0 = 1.0
        # "no" -> incorrect (check_answer=False, check_format=True) -> reward = 0.1*(1-1) + 0.0 = 0.0
        assert total_rewards[0] == pytest.approx(1.0)
        assert total_rewards[1] == pytest.approx(0.0)
        assert total_rewards[2] == pytest.approx(1.0)

    def test_logprobs_preserved(self):
        """Logprobs from the sampling response are correctly threaded into transitions."""
        renderer = MockRenderer()
        G = 2
        logprobs_0 = [-0.1, -0.2, -0.3]
        logprobs_1 = [-0.4, -0.5]
        responses = [
            (_tokens("abc"), logprobs_0),
            (_tokens("de"), logprobs_1),
        ]

        client = _make_fake_sampling_client([responses])
        builder = ProblemGroupBuilder(
            env_thunk=partial(
                SimpleProblemEnv, question="Q", answer="abc", renderer=renderer
            ),
            num_envs=G,
        )

        sampling_params = tinker.SamplingParams(
            stop=["<stop>"], max_tokens=100, temperature=1.0
        )
        result = _run(do_batched_group_rollout(builder, client, sampling_params))

        assert result.trajectories_G[0].transitions[0].ac.logprobs == logprobs_0
        assert result.trajectories_G[1].transitions[0].ac.logprobs == logprobs_1

    def test_single_api_call(self):
        """Verify only ONE sample_async call is made (the whole point)."""
        renderer = MockRenderer()
        G = 8
        responses = [(_tokens("x"), _logprobs(1)) for _ in range(G)]

        call_count = 0
        original_responses = [responses]

        async def counting_sample_async(prompt, num_samples, sampling_params, **kw):
            nonlocal call_count
            call_count += 1
            assert num_samples == G, f"Expected num_samples={G}, got {num_samples}"
            return FakeSampleResponse(
                sequences=[
                    FakeSampledSequence(tokens=t, logprobs=lp) for t, lp in responses
                ]
            )

        client = MagicMock(spec=tinker.SamplingClient)
        client.sample_async = counting_sample_async

        builder = ProblemGroupBuilder(
            env_thunk=partial(
                SimpleProblemEnv, question="Q", answer="x", renderer=renderer
            ),
            num_envs=G,
        )
        sampling_params = tinker.SamplingParams(
            stop=["<stop>"], max_tokens=100, temperature=1.0
        )
        _run(do_batched_group_rollout(builder, client, sampling_params))

        assert call_count == 1

    def test_observation_is_shared_prompt(self):
        """All transitions should reference the same prompt observation."""
        renderer = MockRenderer()
        G = 3
        responses = [(_tokens("a"), _logprobs(1)) for _ in range(G)]
        client = _make_fake_sampling_client([responses])

        builder = ProblemGroupBuilder(
            env_thunk=partial(
                SimpleProblemEnv, question="shared_prompt", answer="a", renderer=renderer
            ),
            num_envs=G,
        )
        sampling_params = tinker.SamplingParams(
            stop=["<stop>"], max_tokens=100, temperature=1.0
        )
        result = _run(do_batched_group_rollout(builder, client, sampling_params))

        # All observations should have the same token content
        ob_lengths = [t.transitions[0].ob.length for t in result.trajectories_G]
        assert len(set(ob_lengths)) == 1

    def test_metrics_and_logs_from_env(self):
        """Step metrics and logs from the env are captured in transitions."""
        renderer = MockRenderer()
        G = 2
        responses = [
            (_tokens("yes"), _logprobs(3)),
            (_tokens("no"), _logprobs(2)),
        ]
        client = _make_fake_sampling_client([responses])

        builder = ProblemGroupBuilder(
            env_thunk=partial(
                SimpleProblemEnv, question="Q", answer="yes", renderer=renderer
            ),
            num_envs=G,
        )
        sampling_params = tinker.SamplingParams(
            stop=["<stop>"], max_tokens=100, temperature=1.0
        )
        result = _run(do_batched_group_rollout(builder, client, sampling_params))

        # ProblemEnv.step() sets metrics like "format", "correct", "time/check_answer_s"
        for traj in result.trajectories_G:
            t = traj.transitions[0]
            assert "format" in t.metrics
            assert "correct" in t.metrics

    def test_empty_response_tokens(self):
        """Handle edge case of empty token response (max_tokens=0 or immediate stop)."""
        renderer = MockRenderer()
        G = 2
        responses = [
            ([], []),  # empty response
            (_tokens("a"), _logprobs(1)),
        ]
        client = _make_fake_sampling_client([responses])

        builder = ProblemGroupBuilder(
            env_thunk=partial(
                SimpleProblemEnv, question="Q", answer="a", renderer=renderer
            ),
            num_envs=G,
        )
        sampling_params = tinker.SamplingParams(
            stop=["<stop>"], max_tokens=100, temperature=1.0
        )
        result = _run(do_batched_group_rollout(builder, client, sampling_params))

        assert len(result.trajectories_G) == G
        assert result.trajectories_G[0].transitions[0].ac.tokens == []
        assert result.trajectories_G[1].transitions[0].ac.tokens == _tokens("a")


class TestTrajectoryToDataParity:
    """Verify trajectories from batched rollout work with the training data pipeline."""

    def test_trajectory_to_data_produces_valid_datums(self):
        """Trajectories from batched rollout can be converted to training data."""
        from tinker_cookbook.rl.data_processing import trajectory_to_data, assemble_training_data

        renderer = MockRenderer()
        G = 3
        answer_texts = ["yes", "no", "yes"]
        responses = [
            (_tokens(text), _logprobs(len(text))) for text in answer_texts
        ]
        client = _make_fake_sampling_client([responses])

        builder = ProblemGroupBuilder(
            env_thunk=partial(
                SimpleProblemEnv, question="Q", answer="yes", renderer=renderer
            ),
            num_envs=G,
        )
        sampling_params = tinker.SamplingParams(
            stop=["<stop>"], max_tokens=100, temperature=1.0
        )
        traj_group = _run(do_batched_group_rollout(builder, client, sampling_params))

        # Each trajectory should produce exactly 1 datum (single-turn)
        for traj in traj_group.trajectories_G:
            datums = trajectory_to_data(traj, traj_advantage=1.0)
            assert len(datums) == 1
            datum = datums[0]
            # Datum should have the expected keys
            assert "target_tokens" in datum.loss_fn_inputs
            assert "logprobs" in datum.loss_fn_inputs
            assert "advantages" in datum.loss_fn_inputs
            assert "mask" in datum.loss_fn_inputs

    def test_assemble_training_data_from_batched_rollout(self):
        """Full pipeline: batched rollout -> advantages -> training data."""
        import torch
        from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages

        renderer = MockRenderer()
        G = 4
        answer_texts = ["yes", "no", "yes", "no"]
        responses = [
            (_tokens(text), _logprobs(len(text))) for text in answer_texts
        ]
        client = _make_fake_sampling_client([responses])

        builder = ProblemGroupBuilder(
            env_thunk=partial(
                SimpleProblemEnv, question="Q", answer="yes", renderer=renderer
            ),
            num_envs=G,
        )
        sampling_params = tinker.SamplingParams(
            stop=["<stop>"], max_tokens=100, temperature=1.0
        )
        traj_group = _run(do_batched_group_rollout(builder, client, sampling_params))

        # Compute advantages
        advantages_P = compute_advantages([traj_group])
        assert len(advantages_P) == 1
        assert len(advantages_P[0]) == G

        # Assemble training data
        data_D, metadata_D = assemble_training_data([traj_group], advantages_P)
        assert len(data_D) == G  # one datum per trajectory (single-turn)
        assert len(metadata_D) == G

        # Verify advantages are non-zero (mixed rewards)
        total_rewards = traj_group.get_total_rewards()
        assert not all(r == total_rewards[0] for r in total_rewards)


class TestBatchedVsUnbatchedParity:
    """Verify that batched and unbatched rollouts produce equivalent results
    when given the same sampled tokens."""

    def test_rewards_match(self):
        """Total rewards from batched rollout match what single rollouts would give."""
        renderer = MockRenderer()
        G = 4
        answer_texts = ["yes", "no", "yes", "maybe"]

        # For batched: one call with G sequences
        batched_responses = [
            [(_tokens(text), _logprobs(len(text))) for text in answer_texts]
        ]
        client_batched = _make_fake_sampling_client(batched_responses)

        builder = ProblemGroupBuilder(
            env_thunk=partial(
                SimpleProblemEnv, question="Q", answer="yes", renderer=renderer
            ),
            num_envs=G,
        )
        sampling_params = tinker.SamplingParams(
            stop=["<stop>"], max_tokens=100, temperature=1.0
        )
        batched_result = _run(
            do_batched_group_rollout(builder, client_batched, sampling_params)
        )

        # Compute expected rewards by running envs directly
        expected_rewards = []
        for text in answer_texts:
            env = SimpleProblemEnv(question="Q", answer="yes", renderer=renderer)

            async def grade_one(env, text):
                await env.initial_observation()
                step_result = await env.step(_tokens(text))
                return step_result.reward

            reward = _run(grade_one(env, text))
            expected_rewards.append(reward)

        batched_total = batched_result.get_total_rewards()
        for i in range(G):
            assert batched_total[i] == pytest.approx(expected_rewards[i]), (
                f"Mismatch at index {i}: batched={batched_total[i]}, expected={expected_rewards[i]}"
            )
