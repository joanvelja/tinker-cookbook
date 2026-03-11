"""Tests for the debate env and builders."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Mapping
from unittest.mock import MagicMock

import pytest
import tinker

from tinker_cookbook.recipes.multiplayer_rl.debate.env import DebateEnv
from tinker_cookbook.recipes.multiplayer_rl.debate.builders import (
    DebateBranchGroupBuilder,
    DebateGroupBuilder,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.dataset import DebateDataset
from tinker_cookbook.recipes.multiplayer_rl.debate.core.runtime import DebateRuntime
from tinker_cookbook.recipes.multiplayer_rl.debate.core.schedule import build_schedule
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateGameSpec,
    DebateOutcome,
    DebateProblemSpec,
    DebateSnapshot,
    DebateSpec,
    DebateState,
    JudgeDecision,
    JudgeRequest,
    ProtocolKind,
    Role,
    ScoringMode,
)
from tinker_cookbook.renderers import Message


# Convenience helpers for test readability.
_MCQ_PROBLEM = DebateProblemSpec.from_seat_answers("Q", "A", "B", ScoringMode.MCQ)
_SEQ_1 = DebateGameSpec(protocol_kind=ProtocolKind.SEQUENTIAL, num_rounds=1)
_SEQ_2 = DebateGameSpec(protocol_kind=ProtocolKind.SEQUENTIAL, num_rounds=2)
_SIM_2 = DebateGameSpec(protocol_kind=ProtocolKind.SIMULTANEOUS, num_rounds=2)


# --- DebateGroupBuilder ---


def test_group_builder_make_envs_count():
    """make_envs produces one env per role in include_roles."""
    builder = DebateGroupBuilder(
        problem=DebateProblemSpec.from_seat_answers("test", "A", "B", ScoringMode.MCQ),
        game=_SEQ_1,
        renderer=MagicMock(),
    )
    envs = asyncio.get_event_loop().run_until_complete(builder.make_envs())
    assert len(envs) == 2  # default: both debaters
    assert all(isinstance(e, DebateEnv) for e in envs)


def test_group_builder_partial_roles_rejected():
    """Partial include_roles that doesn't cover all schedule actors raises ValueError."""
    builder = DebateGroupBuilder(
        problem=_MCQ_PROBLEM,
        game=_SEQ_1,
        renderer=MagicMock(),
        include_roles=(Role.DEBATER_A,),
    )
    with pytest.raises(ValueError, match="Schedule requires roles"):
        asyncio.get_event_loop().run_until_complete(builder.make_envs())


def test_group_builder_logging_tags():
    builder = DebateGroupBuilder(
        problem=_MCQ_PROBLEM,
        game=DebateGameSpec(protocol_kind=ProtocolKind.SIMULTANEOUS, num_rounds=1),
        renderer=MagicMock(),
    )
    assert builder.logging_tags() == ["debate", "simultaneous"]


def test_group_builder_compute_rewards_no_outcome_fn():
    """Without outcome_reward_fn, returns zeros."""
    builder = DebateGroupBuilder(
        problem=_MCQ_PROBLEM,
        game=_SEQ_1,
        renderer=MagicMock(),
    )
    envs = asyncio.get_event_loop().run_until_complete(builder.make_envs())
    # Fake trajectories
    fake_trajs = [MagicMock() for _ in envs]
    rewards = asyncio.get_event_loop().run_until_complete(
        builder.compute_group_rewards(fake_trajs, envs)
    )
    assert all(r == (0.0, {}) for r in rewards)


def test_group_builder_compute_rewards_with_outcome_fn():
    """With outcome_reward_fn, returns mapped rewards."""
    def outcome_fn(outcome: DebateOutcome) -> Mapping[Role, float]:
        return outcome.scores_by_role

    builder = DebateGroupBuilder(
        problem=_MCQ_PROBLEM,
        game=_SEQ_1,
        renderer=MagicMock(),
        outcome_reward_fn=outcome_fn,
    )
    envs = asyncio.get_event_loop().run_until_complete(builder.make_envs())
    assert len(builder._runtimes) == 1

    # Set outcome on state.
    outcome = DebateOutcome(
        winner=Role.DEBATER_A,
        scores_by_role={Role.DEBATER_A: 1.0, Role.DEBATER_B: -1.0},
    )
    builder._runtimes[0]._state = replace(builder._runtimes[0]._state, outcome=outcome)

    fake_trajs = [MagicMock() for _ in envs]
    rewards = asyncio.get_event_loop().run_until_complete(
        builder.compute_group_rewards(fake_trajs, envs)
    )
    # Find which env has which role
    for env, (reward, metrics) in zip(envs, rewards):
        assert isinstance(env, DebateEnv)
        if env.role == Role.DEBATER_A:
            assert reward == 1.0
        else:
            assert reward == -1.0


# --- DebateBranchGroupBuilder ---


def test_branch_builder_creates_independent_envs():
    """Branch builder creates envs from snapshot with independent state."""
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, 2)
    spec = DebateSpec(
        debate_id="test",
        problem=DebateProblemSpec.from_seat_answers("test", "A", "B", ScoringMode.MCQ),
        schedule=schedule,
        open_reasoning=False,
    )
    state = DebateState(
        spec=spec,
        slot_index=0,
        rounds_completed=0,
        transcript=(),
        pending_simultaneous={},
        judge_trace=(),
        done=False,
        outcome=None,
    )
    snapshot = DebateSnapshot(
        state=state,
        renderer_name="test",
    )
    builder = DebateBranchGroupBuilder(
        snapshot=snapshot,
        renderer=MagicMock(),
    )
    envs = asyncio.get_event_loop().run_until_complete(builder.make_envs())
    assert len(envs) == 2
    assert all(isinstance(e, DebateEnv) for e in envs)
    assert builder.logging_tags() == ["debate", "branch", "sequential"]


# --- DebateDataset ---


def test_debate_dataset_len():
    problems = [
        DebateProblemSpec.from_seat_answers("q1", "a1", "b1", ScoringMode.MCQ),
        DebateProblemSpec.from_seat_answers("q2", "a2", "b2", ScoringMode.MCQ),
        DebateProblemSpec.from_seat_answers("q3", "a3", "b3", ScoringMode.MCQ),
    ]
    ds = DebateDataset(
        problems=problems,
        batch_size=2,
        group_size=1,
        game=_SEQ_1,
        renderer=MagicMock(),
    )
    assert len(ds) == 2  # ceil(3/2)


def test_debate_dataset_get_batch():
    problems = [
        DebateProblemSpec.from_seat_answers("q1", "a1", "b1", ScoringMode.MCQ),
        DebateProblemSpec.from_seat_answers("q2", "a2", "b2", ScoringMode.MCQ),
    ]
    ds = DebateDataset(
        problems=problems,
        batch_size=2,
        group_size=1,
        game=_SEQ_1,
        renderer=MagicMock(),
    )
    batch = ds.get_batch(0)
    assert len(batch) == 2
    assert all(isinstance(b, DebateGroupBuilder) for b in batch)
    # Check that task prompts are correct
    assert batch[0].problem.task_prompt == "q1"
    assert batch[1].problem.task_prompt == "q2"


# --- Mocks for frozen-opponent tests ---


class _MockTokenizer:
    """Mock tokenizer matching MockRenderer's char-per-token convention."""

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


class MockCompleter:
    """Mock MessageCompleter that returns a fixed reply."""

    def __init__(self) -> None:
        self.call_count = 0

    async def __call__(self, messages: list[Message]) -> Message:
        self.call_count += 1
        return Message(role="assistant", content=f"opponent turn {self.call_count}")


class MockJudge:
    """Mock JudgeCallback for testing."""

    async def on_boundary(self, request: JudgeRequest) -> JudgeDecision | None:
        return JudgeDecision(
            round_index=request.state.rounds_completed - 1,
            verdict="A wins",
            score_delta_by_role={Role.DEBATER_A: 1.0, Role.DEBATER_B: -1.0},
        )

    async def on_final(self, request: JudgeRequest) -> DebateOutcome:
        return DebateOutcome(
            winner=Role.DEBATER_A,
            scores_by_role={Role.DEBATER_A: 1.0, Role.DEBATER_B: 0.0},
        )


# --- Frozen-opponent tests ---


def _run(coro):
    """Run a coroutine in the event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _rollout(env: DebateEnv) -> list:
    """Drive a single env to completion, returning step results."""

    async def _go():
        await env.initial_observation()
        results = []
        while True:
            tokens = [ord(c) for c in f"I am {env.role.value}"]
            result = await env.step(tokens)
            results.append(result)
            if result.episode_done:
                break
        return results

    return _run(_go())


def test_frozen_opponent_sequential():
    """A trains, B frozen. Verify alternation and completion."""
    renderer = MockRenderer()
    completer = MockCompleter()
    builder = DebateGroupBuilder(
        problem=_MCQ_PROBLEM,
        game=_SEQ_2,
        renderer=renderer,
        opponent_completer=completer,
        group_size=1,
    )
    envs = _run(builder.make_envs())
    assert len(envs) == 1
    env = envs[0]
    assert isinstance(env, DebateEnv)
    assert env.role == Role.DEBATER_A
    assert env.opponent_role == Role.DEBATER_B

    results = _rollout(env)
    assert results[-1].episode_done
    assert env.runtime.state.done
    assert len(env.runtime.state.transcript) == 4
    # Opponent was called twice (once per round, B responds after A).
    assert completer.call_count == 2


def test_frozen_opponent_b_first():
    """Trained=DEBATER_B, verify B waits for frozen A to go first."""
    renderer = MockRenderer()
    completer = MockCompleter()
    builder = DebateGroupBuilder(
        problem=_MCQ_PROBLEM,
        game=_SEQ_1,
        renderer=MagicMock(),
    )
    # Manually set up frozen-opponent with B as the trained role.
    # We bypass the builder to control role assignment.
    _run(builder.make_envs())  # triggers schedule build
    from tinker_cookbook.recipes.multiplayer_rl.debate.core.schedule import build_schedule as bs

    schedule = bs(ProtocolKind.SEQUENTIAL, 1)
    spec = DebateSpec(
        debate_id="test-b-first",
        problem=DebateProblemSpec.from_seat_answers("Q", "A", "B", ScoringMode.MCQ),
        schedule=schedule,
        open_reasoning=False,
    )
    state = DebateState(
        spec=spec,
        slot_index=0,
        rounds_completed=0,
        transcript=(),
        pending_simultaneous={},
        judge_trace=(),
        done=False,
        outcome=None,
    )
    runtime = DebateRuntime(state)
    env = DebateEnv(
        role=Role.DEBATER_B,
        runtime=runtime,
        renderer=renderer,
        opponent_completer=completer,
        opponent_role=Role.DEBATER_A,
    )
    results = _rollout(env)
    assert results[-1].episode_done
    assert runtime.state.done
    # A goes first (frozen), then B, so transcript should be A, B.
    assert runtime.state.transcript[0].role == Role.DEBATER_A
    assert runtime.state.transcript[1].role == Role.DEBATER_B


def test_frozen_opponent_hybrid():
    """Hybrid: simultaneous round-0 proposals + sequential critiques."""
    renderer = MockRenderer()
    completer = MockCompleter()

    # Hybrid requires num_rounds >= 2
    from tinker_cookbook.recipes.multiplayer_rl.debate.core.schedule import build_schedule as bs

    schedule = bs(ProtocolKind.HYBRID, 2)
    spec = DebateSpec(
        debate_id="hybrid-frozen",
        problem=DebateProblemSpec.from_seat_answers("Q", "A", "B", ScoringMode.MCQ),
        schedule=schedule,
        open_reasoning=False,
    )
    state = DebateState(
        spec=spec,
        slot_index=0,
        rounds_completed=0,
        transcript=(),
        pending_simultaneous={},
        judge_trace=(),
        done=False,
        outcome=None,
    )
    runtime = DebateRuntime(state)
    env = DebateEnv(
        role=Role.DEBATER_A,
        runtime=runtime,
        renderer=renderer,
        opponent_completer=completer,
        opponent_role=Role.DEBATER_B,
    )

    async def _go():
        await env.initial_observation()
        results = []
        while True:
            tokens = [ord(c) for c in f"trained {env.role.value}"]
            result = await env.step(tokens)
            results.append(result)
            if result.episode_done:
                break
        return results

    results = _run(_go())
    assert results[-1].episode_done
    assert runtime.state.done
    assert len(runtime.state.transcript) == 4


def test_frozen_opponent_group_size():
    """group_size=4 creates 4 independent runtimes."""
    renderer = MockRenderer()
    completer = MockCompleter()
    builder = DebateGroupBuilder(
        problem=_MCQ_PROBLEM,
        game=_SEQ_1,
        renderer=renderer,
        opponent_completer=completer,
        group_size=4,
    )
    envs = _run(builder.make_envs())
    assert len(envs) == 4
    assert len(builder._runtimes) == 4
    # Each env has its own runtime.
    runtimes = [e.runtime for e in envs]  # type: ignore[union-attr]
    assert len(set(id(r) for r in runtimes)) == 4


def test_frozen_opponent_randomize_position():
    """Over 100 builds, ~50/50 role assignment."""
    renderer = MockRenderer()
    completer = MockCompleter()
    roles = []
    for _ in range(100):
        builder = DebateGroupBuilder(
            problem=_MCQ_PROBLEM,
            game=_SEQ_1,
            renderer=renderer,
            opponent_completer=completer,
            group_size=1,
            randomize_position=True,
        )
        envs = _run(builder.make_envs())
        assert len(envs) == 1
        env = envs[0]
        assert isinstance(env, DebateEnv)
        roles.append(env.role)

    a_count = sum(1 for r in roles if r == Role.DEBATER_A)
    b_count = sum(1 for r in roles if r == Role.DEBATER_B)
    # With 100 trials, the probability of getting < 20 or > 80 of either is vanishingly small.
    assert 20 < a_count < 80, f"Expected ~50/50 split, got A={a_count}, B={b_count}"


def test_frozen_opponent_compute_group_rewards():
    """Correct per-role reward mapping in frozen-opponent mode."""
    renderer = MockRenderer()
    completer = MockCompleter()

    def outcome_fn(outcome: DebateOutcome) -> Mapping[Role, float]:
        return outcome.scores_by_role

    builder = DebateGroupBuilder(
        problem=_MCQ_PROBLEM,
        game=_SEQ_1,
        renderer=renderer,
        opponent_completer=completer,
        group_size=2,
        outcome_reward_fn=outcome_fn,
        judge_callback=MockJudge(),
    )
    envs = _run(builder.make_envs())
    assert len(envs) == 2

    # Run all envs to completion.
    async def _run_all():
        for env in envs:
            assert isinstance(env, DebateEnv)
            await env.initial_observation()
            while True:
                tokens = [ord(c) for c in f"trained {env.role.value}"]
                result = await env.step(tokens)
                if result.episode_done:
                    break

    _run(_run_all())

    fake_trajs = [MagicMock() for _ in envs]
    rewards = _run(builder.compute_group_rewards(fake_trajs, envs))
    for env, (reward, _) in zip(envs, rewards):
        assert isinstance(env, DebateEnv)
        # MockJudge always declares A the winner with scores {A: 1.0, B: 0.0}.
        expected = 1.0 if env.role == Role.DEBATER_A else 0.0
        assert reward == expected


def test_frozen_opponent_uses_opponent_renderer_tokenizer():
    """When opponent_renderer is set, _opponent_submit counts tokens with its tokenizer."""

    class _WordTokenizer:
        """Tokenizer that splits on whitespace (1 token per word)."""

        def encode(self, text: str) -> list[int]:
            return list(range(len(text.split())))

        def decode(self, tokens: list[int]) -> str:
            return " ".join(str(t) for t in tokens)

    class WordRenderer(MockRenderer):
        def __init__(self) -> None:
            self.tokenizer = _WordTokenizer()

    # trained renderer: char-per-token (e.g. "hello" = 5 tokens)
    trained_renderer = MockRenderer()
    # opponent renderer: word-per-token (e.g. "opponent turn 1" = 3 tokens)
    opponent_renderer = WordRenderer()
    completer = MockCompleter()

    builder = DebateGroupBuilder(
        problem=_MCQ_PROBLEM,
        game=_SEQ_1,
        renderer=trained_renderer,
        opponent_completer=completer,
        opponent_renderer=opponent_renderer,
        group_size=1,
    )
    envs = _run(builder.make_envs())
    env = envs[0]
    assert isinstance(env, DebateEnv)
    _rollout(env)

    # The opponent said "opponent turn 1" (MockCompleter). With the word tokenizer
    # that's 3 tokens. With the char tokenizer it would be 15 tokens.
    opponent_utterances = [u for u in env.runtime.state.transcript if u.role != env.role]
    assert len(opponent_utterances) == 1
    assert opponent_utterances[0].token_count == 3  # word tokenizer, not char


def test_frozen_opponent_fallback_to_trained_renderer():
    """Without opponent_renderer, token count uses trained renderer's tokenizer."""
    renderer = MockRenderer()
    completer = MockCompleter()

    builder = DebateGroupBuilder(
        problem=_MCQ_PROBLEM,
        game=_SEQ_1,
        renderer=renderer,
        opponent_completer=completer,
        group_size=1,
    )
    envs = _run(builder.make_envs())
    env = envs[0]
    assert isinstance(env, DebateEnv)
    _rollout(env)

    opponent_utterances = [u for u in env.runtime.state.transcript if u.role != env.role]
    assert len(opponent_utterances) == 1
    # "opponent turn 1" = 15 chars = 15 tokens with MockRenderer's char tokenizer
    assert opponent_utterances[0].token_count == len("opponent turn 1")


def test_frozen_opponent_simultaneous():
    """Pure SIMULTANEOUS works with frozen opponent via ensure_future barrier."""
    renderer = MockRenderer()
    completer = MockCompleter()
    builder = DebateGroupBuilder(
        problem=_MCQ_PROBLEM,
        game=_SIM_2,
        renderer=renderer,
        opponent_completer=completer,
        group_size=1,
    )
    envs = _run(builder.make_envs())
    assert len(envs) == 1
    env = envs[0]
    assert isinstance(env, DebateEnv)

    results = _rollout(env)
    assert results[-1].episode_done
    assert env.runtime.state.done
    assert len(env.runtime.state.transcript) == 4  # 2 rounds x 2 debaters
    assert completer.call_count == 2


# --- Self-play tests ---


def test_selfplay_make_envs():
    """Self-play with group_size=3 creates 6 envs (3 runtimes x 2 roles), pairs share runtime."""
    renderer = MockRenderer()
    builder = DebateGroupBuilder(
        problem=_MCQ_PROBLEM,
        game=_SEQ_1,
        renderer=renderer,
        group_size=3,
    )
    envs = _run(builder.make_envs())
    assert len(envs) == 6  # 3 runtimes x 2 roles
    assert len(builder._runtimes) == 3

    # Each pair of envs shares a runtime.
    for i in range(3):
        env_a = envs[2 * i]
        env_b = envs[2 * i + 1]
        assert isinstance(env_a, DebateEnv)
        assert isinstance(env_b, DebateEnv)
        assert env_a.runtime is env_b.runtime
        assert {env_a.role, env_b.role} == {Role.DEBATER_A, Role.DEBATER_B}

    # All runtimes are distinct.
    runtimes = [e.runtime for e in envs if isinstance(e, DebateEnv)]
    runtime_ids = [id(runtimes[0]), id(runtimes[2]), id(runtimes[4])]
    assert len(set(runtime_ids)) == 3


def test_selfplay_rollout():
    """Self-play rollout: both agents take turns concurrently via gather."""
    renderer = MockRenderer()
    builder = DebateGroupBuilder(
        problem=_MCQ_PROBLEM,
        game=_SEQ_1,
        renderer=renderer,
        group_size=1,
        judge_callback=MockJudge(),
    )
    envs = _run(builder.make_envs())
    assert len(envs) == 2
    env_a, env_b = envs
    assert isinstance(env_a, DebateEnv)
    assert isinstance(env_b, DebateEnv)
    assert env_a.runtime is env_b.runtime

    async def _drive(env: DebateEnv) -> list:
        await env.initial_observation()
        results = []
        while True:
            tokens = [ord(c) for c in f"I am {env.role.value}"]
            result = await env.step(tokens)
            results.append(result)
            if result.episode_done:
                break
        return results

    async def _run_both():
        return await asyncio.gather(_drive(env_a), _drive(env_b))

    results_a, results_b = _run(_run_both())
    assert results_a[-1].episode_done
    assert results_b[-1].episode_done
    runtime = env_a.runtime
    assert runtime.state.done
    # Sequential 1-round: A then B = 2 utterances.
    assert len(runtime.state.transcript) == 2
    assert runtime.state.transcript[0].role == Role.DEBATER_A
    assert runtime.state.transcript[1].role == Role.DEBATER_B


def test_selfplay_compute_group_rewards():
    """Self-play rewards are zero-sum pairs, no identity remapping."""
    renderer = MockRenderer()

    def outcome_fn(outcome: DebateOutcome) -> Mapping[Role, float]:
        return outcome.scores_by_role

    builder = DebateGroupBuilder(
        problem=_MCQ_PROBLEM,
        game=_SEQ_1,
        renderer=renderer,
        group_size=2,
        outcome_reward_fn=outcome_fn,
        judge_callback=MockJudge(),
    )
    envs = _run(builder.make_envs())
    assert len(envs) == 4  # 2 runtimes x 2 roles

    async def _drive(env: DebateEnv) -> None:
        await env.initial_observation()
        while True:
            tokens = [ord(c) for c in f"I am {env.role.value}"]
            result = await env.step(tokens)
            if result.episode_done:
                break

    async def _run_all():
        # Gather per-runtime pairs concurrently.
        tasks = []
        for i in range(0, len(envs), 2):
            tasks.append(asyncio.gather(_drive(envs[i]), _drive(envs[i + 1])))
        await asyncio.gather(*tasks)

    _run(_run_all())

    fake_trajs = [MagicMock() for _ in envs]
    rewards = _run(builder.compute_group_rewards(fake_trajs, envs))

    # MockJudge: A wins with {A: 1.0, B: 0.0}.
    for env, (reward, metrics) in zip(envs, rewards):
        assert isinstance(env, DebateEnv)
        expected = 1.0 if env.role == Role.DEBATER_A else 0.0
        assert reward == expected
        # Self-play mode: no identity remap (opponent_completer is None).
        assert not any(k.startswith("id/") for k in metrics)


def test_selfplay_randomize_position_guard():
    """randomize_position=True raises ValueError in self-play mode."""
    renderer = MockRenderer()
    builder = DebateGroupBuilder(
        problem=_MCQ_PROBLEM,
        game=_SEQ_1,
        renderer=renderer,
        randomize_position=True,
        # No opponent_completer -> self-play mode.
    )
    with pytest.raises(ValueError, match="randomize_position"):
        _run(builder.make_envs())
