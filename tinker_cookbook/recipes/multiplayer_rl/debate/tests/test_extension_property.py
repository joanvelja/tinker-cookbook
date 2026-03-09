"""Regression test: consecutive observations must be prefix-monotonic.

The extension property ensures that each turn's message list is a strict prefix
of the next turn's message list (for the same viewer). This enables O(T) GPU
prefill via KV-cache reuse instead of O(T^2) re-encoding from scratch.
"""

from __future__ import annotations

import pytest

from tinker_cookbook.recipes.multiplayer_rl.debate.core.runtime import DebateRuntime
from tinker_cookbook.recipes.multiplayer_rl.debate.core.schedule import build_schedule
from tinker_cookbook.recipes.multiplayer_rl.debate.core.visibility import build_generation_messages
from tinker_cookbook.recipes.multiplayer_rl.debate.tests.conftest import make_spec
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateState,
    ProtocolKind,
    Role,
    ScoringMode,
)


def _initial_state(num_rounds: int = 2, prompts_ref: str = "default") -> DebateState:
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, num_rounds)
    spec = make_spec(
        task_prompt="What is 2+2?",
        scoring_mode=ScoringMode.MCQ,
        answer_by_role={Role.DEBATER_A: "4", Role.DEBATER_B: "5"},
        schedule=schedule,
        open_reasoning=False,
        prompts_ref=prompts_ref,
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


def _assert_prefix(earlier: list, later: list, turn_a: int, turn_b: int) -> None:
    """Assert that `earlier` messages are a strict prefix of `later`."""
    assert len(later) > len(earlier), (
        f"Observation at turn {turn_b} should be longer than turn {turn_a}: "
        f"{len(later)} vs {len(earlier)}"
    )
    for j, (e_msg, l_msg) in enumerate(zip(earlier, later)):
        assert e_msg == l_msg, (
            f"Prefix mismatch at message {j} between turn {turn_a} and {turn_b}:\n"
            f"  earlier: {e_msg}\n"
            f"  later:   {l_msg}"
        )


@pytest.mark.asyncio
async def test_extension_property_sequential():
    """Consecutive observations for a debater share a common prefix (sequential protocol)."""
    state = _initial_state(num_rounds=2)
    runtime = DebateRuntime(state)
    observations: list[list] = []

    role_order = [Role.DEBATER_A, Role.DEBATER_B]
    for turn in range(4):  # 2 rounds x 2 debaters
        role = role_order[turn % 2]
        ticket = await runtime.wait_for_turn(role)
        if ticket is None:
            break

        if role == Role.DEBATER_A:
            msgs, _ = build_generation_messages(runtime.state, Role.DEBATER_A)
            observations.append(msgs)

        text = f"<think>reasoning turn {turn}</think>Argument turn {turn}"
        await runtime.submit(ticket, text, len(text))

    assert len(observations) >= 2, f"Expected >=2 observations, got {len(observations)}"
    for i in range(len(observations) - 1):
        _assert_prefix(observations[i], observations[i + 1], i, i + 1)


@pytest.mark.asyncio
async def test_extension_property_preserves_thinking():
    """Own-turn thinking tokens are preserved in historical messages (not stripped)."""
    state = _initial_state(num_rounds=2)
    runtime = DebateRuntime(state)
    think_text = "<think>my secret plan</think>My public argument"

    # Turn 0: Debater A
    ticket = await runtime.wait_for_turn(Role.DEBATER_A)
    assert ticket is not None
    await runtime.submit(ticket, think_text, len(think_text))

    # Turn 1: Debater B
    ticket = await runtime.wait_for_turn(Role.DEBATER_B)
    assert ticket is not None
    await runtime.submit(ticket, "B's argument", 13)

    # Turn 2: Debater A again — check that their first turn retains thinking
    msgs, _ = build_generation_messages(runtime.state, Role.DEBATER_A)
    assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
    assert len(assistant_msgs) >= 1
    assert "<think>my secret plan</think>" in assistant_msgs[0]["content"], (
        "Own thinking must be preserved for KV-cache prefix reuse"
    )


@pytest.mark.asyncio
async def test_extension_property_strips_opponent_thinking():
    """Opponent thinking is still stripped even though own thinking is preserved."""
    state = _initial_state(num_rounds=1)
    runtime = DebateRuntime(state)

    # Turn 0: Debater A with thinking
    ticket = await runtime.wait_for_turn(Role.DEBATER_A)
    assert ticket is not None
    await runtime.submit(ticket, "<think>A secret</think>A public", 30)

    # Turn 1: Debater B with thinking
    ticket = await runtime.wait_for_turn(Role.DEBATER_B)
    assert ticket is not None

    # Before B submits, check B's view: A's thinking should be stripped
    msgs, _ = build_generation_messages(runtime.state, Role.DEBATER_B)
    opponent_msgs = [m for m in msgs if m["role"] == "user" and "opponent_turn" in m.get("content", "")]
    assert len(opponent_msgs) >= 1
    assert "<think>" not in opponent_msgs[0]["content"], (
        "Opponent thinking must be stripped when open_reasoning=False"
    )


@pytest.mark.asyncio
async def test_extension_property_selfplay_prompts():
    """Extension property holds with selfplay prompts (which have non-empty user templates)."""
    state = _initial_state(num_rounds=2, prompts_ref="selfplay")
    runtime = DebateRuntime(state)
    observations: list[list] = []

    role_order = [Role.DEBATER_A, Role.DEBATER_B]
    for turn in range(4):
        role = role_order[turn % 2]
        ticket = await runtime.wait_for_turn(role)
        if ticket is None:
            break

        if role == Role.DEBATER_A:
            msgs, _ = build_generation_messages(runtime.state, Role.DEBATER_A)
            observations.append(msgs)

        text = f"<think>reasoning {turn}</think><answer>4</answer>Argument {turn}"
        await runtime.submit(ticket, text, len(text))

    assert len(observations) >= 2, f"Expected >=2 observations, got {len(observations)}"
    for i in range(len(observations) - 1):
        _assert_prefix(observations[i], observations[i + 1], i, i + 1)


@pytest.mark.asyncio
async def test_extension_property_gpqa_open_prompts():
    """Extension property holds with open_balanced prompts (non-empty user templates, no stance)."""
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, 2)
    spec = make_spec(
        task_prompt="What is the mechanism of X?",
        scoring_mode=ScoringMode.OPEN_ENDED,
        answer_by_role=None,
        target="Y",
        schedule=schedule,
        prompts_ref="open_balanced",
    )
    state = DebateState(
        spec=spec, slot_index=0, rounds_completed=0,
        transcript=(), pending_simultaneous={},
        judge_trace=(), done=False, outcome=None,
    )
    runtime = DebateRuntime(state)
    observations: list[list] = []

    role_order = [Role.DEBATER_A, Role.DEBATER_B]
    for turn in range(4):
        role = role_order[turn % 2]
        ticket = await runtime.wait_for_turn(role)
        if ticket is None:
            break

        if role == Role.DEBATER_A:
            msgs, _ = build_generation_messages(runtime.state, Role.DEBATER_A)
            observations.append(msgs)

        text = f"<answer>answer_{turn}</answer><reasoning>because {turn}</reasoning>Argument {turn}"
        await runtime.submit(ticket, text, len(text))

    assert len(observations) >= 2, f"Expected >=2 observations, got {len(observations)}"
    for i in range(len(observations) - 1):
        _assert_prefix(observations[i], observations[i + 1], i, i + 1)

    # Verify instructions are actually interleaved (not just empty)
    last_msgs = observations[-1]
    user_contents = [m["content"] for m in last_msgs if m["role"] == "user"]
    all_user_text = "\n".join(user_contents)
    assert "State your answer" in all_user_text, "propose instruction should be interleaved"
    assert "Identify the most important" in all_user_text, "critique instruction should be interleaved"
