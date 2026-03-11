"""Tests for the debate runtime (async coordination)."""

from __future__ import annotations

import asyncio

import pytest

from tinker_cookbook.recipes.multiplayer_rl.debate.core.runtime import DebateRuntime
from tinker_cookbook.recipes.multiplayer_rl.debate.core.schedule import build_schedule
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateOutcome,
    DebateProblemSpec,
    DebateSpec,
    DebateState,
    JudgeDecision,
    JudgeRequest,
    ProtocolKind,
    Role,
    ScoringMode,
)


def _make_state(
    kind: ProtocolKind = ProtocolKind.SEQUENTIAL,
    num_rounds: int = 2,
) -> DebateState:
    schedule = build_schedule(kind, num_rounds)
    spec = DebateSpec(
        debate_id="test",
        problem=DebateProblemSpec(
            task_prompt="Which is bigger?",
            scoring_mode=ScoringMode.MCQ,
            answer_by_role={Role.DEBATER_A: "2", Role.DEBATER_B: "3"},
        ),
        schedule=schedule,
        open_reasoning=False,
    )
    return DebateState(
        spec=spec,
        slot_index=0,
        rounds_completed=0,
        transcript=(),
        pending_simultaneous={},
        judge_trace=(),
        done=False,
        outcome=None,
    )


# --- Sequential alternation ---


@pytest.mark.asyncio
async def test_sequential_alternation():
    """Two concurrent coroutines alternate correctly (A, B, A, B)."""
    state = _make_state(kind=ProtocolKind.SEQUENTIAL, num_rounds=2)
    runtime = DebateRuntime(state)
    turns: list[Role] = []

    async def play(role: Role):
        while True:
            ticket = await runtime.wait_for_turn(role)
            if ticket is None:
                break
            result = await runtime.submit(ticket, f"{role.value} speaks", 2)
            turns.append(role)
            if result.episode_done:
                break

    await asyncio.gather(play(Role.DEBATER_A), play(Role.DEBATER_B))
    assert turns == [Role.DEBATER_A, Role.DEBATER_B, Role.DEBATER_A, Role.DEBATER_B]
    assert runtime.state.done
    assert len(runtime.state.transcript) == 4


# --- Simultaneous barrier ---


@pytest.mark.asyncio
async def test_simultaneous_barrier():
    """Both debaters submit; barrier synchronizes correctly."""
    state = _make_state(kind=ProtocolKind.SIMULTANEOUS, num_rounds=1)
    runtime = DebateRuntime(state)

    async def play(role: Role):
        ticket = await runtime.wait_for_turn(role)
        assert ticket is not None
        result = await runtime.submit(ticket, f"{role.value} speaks", 2)
        return result

    results = await asyncio.gather(play(Role.DEBATER_A), play(Role.DEBATER_B))
    # Both should get episode_done=True
    assert all(r.episode_done for r in results)
    assert runtime.state.done
    assert len(runtime.state.transcript) == 2


@pytest.mark.asyncio
async def test_simultaneous_multi_round():
    """Simultaneous with 2 rounds."""
    state = _make_state(kind=ProtocolKind.SIMULTANEOUS, num_rounds=2)
    runtime = DebateRuntime(state)
    turns_a: list[int] = []
    turns_b: list[int] = []

    async def play(role: Role, log: list[int]):
        round_num = 0
        while True:
            ticket = await runtime.wait_for_turn(role)
            if ticket is None:
                break
            result = await runtime.submit(ticket, f"{role.value} r{round_num}", 2)
            log.append(round_num)
            round_num += 1
            if result.episode_done:
                break

    await asyncio.gather(play(Role.DEBATER_A, turns_a), play(Role.DEBATER_B, turns_b))
    assert len(turns_a) == 2
    assert len(turns_b) == 2
    assert runtime.state.done


# --- Stale ticket rejection ---


@pytest.mark.asyncio
async def test_stale_ticket_rejected():
    state = _make_state(kind=ProtocolKind.SEQUENTIAL, num_rounds=1)
    runtime = DebateRuntime(state)

    ticket = await runtime.wait_for_turn(Role.DEBATER_A)
    assert ticket is not None
    # Submit to advance state
    await runtime.submit(ticket, "A speaks", 2)
    # Old ticket is now stale
    with pytest.raises(ValueError, match="Stale ticket"):
        await runtime.submit(ticket, "A speaks again", 2)


# --- CancelledError doesn't deadlock ---


@pytest.mark.asyncio
async def test_cancelled_error_no_deadlock():
    """CancelledError during wait_for_turn must not deadlock other waiters."""
    state = _make_state(kind=ProtocolKind.SEQUENTIAL, num_rounds=1)
    runtime = DebateRuntime(state)

    async def waiter_that_cancels():
        # This waiter will be cancelled while waiting
        await runtime.wait_for_turn(Role.DEBATER_B)

    task_b = asyncio.create_task(waiter_that_cancels())
    # Let B start waiting
    await asyncio.sleep(0.01)
    # Cancel B
    task_b.cancel()
    try:
        await task_b
    except asyncio.CancelledError:
        pass

    # A should still be able to proceed
    ticket = await runtime.wait_for_turn(Role.DEBATER_A)
    assert ticket is not None
    result = await runtime.submit(ticket, "A speaks", 2)
    assert not result.episode_done  # still need B's turn


# --- Step reward ---


@pytest.mark.asyncio
async def test_step_reward():
    def reward_fn(before, after, role, utterance):
        return 1.0 if role == Role.DEBATER_A else -1.0

    state = _make_state(kind=ProtocolKind.SEQUENTIAL, num_rounds=1)
    runtime = DebateRuntime(state, step_reward_fn=reward_fn)

    ticket = await runtime.wait_for_turn(Role.DEBATER_A)
    assert ticket is not None
    result = await runtime.submit(ticket, "A speaks", 2)
    assert result.reward == 1.0


# --- Judge callback ---


@pytest.mark.asyncio
async def test_judge_callback():
    boundary_calls: list[str] = []
    final_calls: list[str] = []

    class MockJudge:
        async def on_boundary(self, request: JudgeRequest) -> JudgeDecision | None:
            boundary_calls.append(request.trigger)
            return JudgeDecision(
                round_index=0,
                verdict="A is better",
                score_delta_by_role={Role.DEBATER_A: 1.0, Role.DEBATER_B: -1.0},
            )

        async def on_final(self, request: JudgeRequest) -> DebateOutcome:
            final_calls.append(request.trigger)
            return DebateOutcome(
                winner=Role.DEBATER_A,
                scores_by_role={Role.DEBATER_A: 1.0, Role.DEBATER_B: 0.0},
            )

    state = _make_state(kind=ProtocolKind.SEQUENTIAL, num_rounds=1)
    runtime = DebateRuntime(state, judge_callback=MockJudge())

    async def play(role: Role):
        while True:
            ticket = await runtime.wait_for_turn(role)
            if ticket is None:
                break
            result = await runtime.submit(ticket, f"{role.value} speaks", 2)
            if result.episode_done:
                break

    await asyncio.gather(play(Role.DEBATER_A), play(Role.DEBATER_B))

    # Boundary called once (after round 0 completes)
    assert len(boundary_calls) == 1
    # Final called once (episode end)
    assert len(final_calls) == 1
    # Judge trace recorded
    assert len(runtime.state.judge_trace) == 1
    # Outcome set
    assert runtime.state.outcome is not None
    assert runtime.state.outcome.winner == Role.DEBATER_A


# --- Snapshot ---


def test_snapshot():
    state = _make_state()
    runtime = DebateRuntime(state)
    snap = runtime.snapshot("llama3")
    assert snap.state is state
    assert snap.renderer_name == "llama3"


# --- Hybrid protocol integration ---


@pytest.mark.asyncio
async def test_hybrid_integration():
    state = _make_state(kind=ProtocolKind.HYBRID, num_rounds=2)
    runtime = DebateRuntime(state)
    turn_order: list[str] = []

    async def play(role: Role):
        while True:
            ticket = await runtime.wait_for_turn(role)
            if ticket is None:
                break
            result = await runtime.submit(ticket, f"{role.value} speaks", 2)
            turn_order.append(role.value)
            if result.episode_done:
                break

    await asyncio.gather(play(Role.DEBATER_A), play(Role.DEBATER_B))
    assert runtime.state.done
    assert len(runtime.state.transcript) == 4
    # Round 0: simultaneous (A+B), Round 1: sequential (A, B)
    # Both A and B submit in round 0, then A then B in round 1
