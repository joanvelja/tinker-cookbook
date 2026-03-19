"""Stress test 6: Full integration — build -> messages -> strip -> correct output.

Tests think_visibility across three prompt packs:
1. open_selfplay_judgesees: debaters visible_to_judge, judge private
2. open_selfplay_private: all thinking private
3. open_selfplay: no debater thinking, judge private
"""
import sys
sys.path.insert(0, ".")

from tinker_cookbook.recipes.multiplayer_rl.debate.prompts import resolve_prompts
from tinker_cookbook.recipes.multiplayer_rl.debate.core.schedule import build_schedule
from tinker_cookbook.recipes.multiplayer_rl.debate.core.visibility import build_generation_messages
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateProblemSpec,
    DebateSpec,
    DebateState,
    Phase,
    ProtocolKind,
    Role,
    ScoringMode,
    ThinkVisibility,
    Utterance,
)


def make_state_with_utterances(prompts_ref: str) -> DebateState:
    """Build a DebateState with 2-round sequential schedule and fake utterances."""
    # Clear LRU cache to avoid stale prompts
    resolve_prompts.cache_clear()

    prompts = resolve_prompts(prompts_ref)
    tv = prompts.get_think_visibility()

    schedule = build_schedule(ProtocolKind.SEQUENTIAL, 2)
    spec = DebateSpec(
        debate_id="stress6",
        problem=DebateProblemSpec(
            task_prompt="What is 2+2?",
            scoring_mode=ScoringMode.OPEN_ENDED,
            answer_by_role={Role.DEBATER_A: "4", Role.DEBATER_B: "5"},
        ),
        schedule=schedule,
        think_visibility=tv,
        protocol_kind=ProtocolKind.SEQUENTIAL,
        prompts_ref=prompts_ref,
    )

    # Add fake utterances from both debaters with thinking tags
    utts = []
    for i, (role, slot_id) in enumerate([
        (Role.DEBATER_A, 0),
        (Role.DEBATER_B, 1),
        (Role.DEBATER_A, 2),
        (Role.DEBATER_B, 3),
    ]):
        phase = Phase.PROPOSE if i < 2 else Phase.CRITIQUE
        round_idx = 0 if i < 2 else 1
        utts.append(Utterance(
            role=role,
            round_index=round_idx,
            phase=phase,
            text=f"<thinking>secret_{role.value}_r{round_idx}</thinking>public_{role.value}_r{round_idx}",
            token_count=20,
            slot_id=slot_id,
        ))

    return DebateState(
        spec=spec,
        slot_index=len(schedule),  # past end = done
        rounds_completed=2,
        transcript=tuple(utts),
        pending_simultaneous={},
        judge_trace=(),
        done=True,
        outcome=None,
    )


def messages_contain(msgs, text):
    """Check if any message content contains the text."""
    for m in msgs:
        content = m.get("content", "")
        if isinstance(content, str) and text in content:
            return True
    return False


def test_open_selfplay_judgesees():
    """Judge sees debater thinking; debaters don't see each other's thinking."""
    print("Testing open_selfplay_judgesees...")
    state = make_state_with_utterances("open_selfplay_judgesees")

    # Verify think_visibility
    tv = state.spec.think_visibility
    assert tv[Role.DEBATER_A] == ThinkVisibility.VISIBLE_TO_JUDGE, f"Expected VISIBLE_TO_JUDGE for A, got {tv[Role.DEBATER_A]}"
    assert tv[Role.DEBATER_B] == ThinkVisibility.VISIBLE_TO_JUDGE, f"Expected VISIBLE_TO_JUDGE for B, got {tv[Role.DEBATER_B]}"
    assert tv[Role.JUDGE] == ThinkVisibility.PRIVATE, f"Expected PRIVATE for Judge, got {tv[Role.JUDGE]}"

    # Judge sees debater A's secret thinking
    judge_msgs, _ = build_generation_messages(state, Role.JUDGE, trigger="final")
    assert messages_contain(judge_msgs, "secret_debater_a"), "Judge should see debater_a's thinking"
    assert messages_contain(judge_msgs, "secret_debater_b"), "Judge should see debater_b's thinking"

    # Debater A does NOT see debater B's secret thinking
    a_msgs, _ = build_generation_messages(state, Role.DEBATER_A, trigger="final")
    assert not messages_contain(a_msgs, "secret_debater_b"), "Debater A should NOT see debater B's thinking"
    # But A sees own thinking (KV-cache)
    assert messages_contain(a_msgs, "secret_debater_a"), "Debater A should see own thinking"

    # Debater B does NOT see debater A's secret thinking
    b_msgs, _ = build_generation_messages(state, Role.DEBATER_B, trigger="final")
    assert not messages_contain(b_msgs, "secret_debater_a"), "Debater B should NOT see debater A's thinking"
    assert messages_contain(b_msgs, "secret_debater_b"), "Debater B should see own thinking"

    print("  PASS: open_selfplay_judgesees")


def test_open_selfplay_private():
    """Nobody sees others' thinking (all private)."""
    print("Testing open_selfplay_private...")
    state = make_state_with_utterances("open_selfplay_private")

    # Verify think_visibility
    tv = state.spec.think_visibility
    assert tv[Role.DEBATER_A] == ThinkVisibility.PRIVATE, f"Expected PRIVATE for A, got {tv[Role.DEBATER_A]}"
    assert tv[Role.DEBATER_B] == ThinkVisibility.PRIVATE, f"Expected PRIVATE for B, got {tv[Role.DEBATER_B]}"
    assert tv[Role.JUDGE] == ThinkVisibility.PRIVATE, f"Expected PRIVATE for Judge, got {tv[Role.JUDGE]}"

    # Judge does NOT see debater thinking
    judge_msgs, _ = build_generation_messages(state, Role.JUDGE, trigger="final")
    assert not messages_contain(judge_msgs, "secret_debater_a"), "Judge should NOT see debater_a's thinking (private)"
    assert not messages_contain(judge_msgs, "secret_debater_b"), "Judge should NOT see debater_b's thinking (private)"

    # Debater A does NOT see debater B's thinking, but sees own
    a_msgs, _ = build_generation_messages(state, Role.DEBATER_A, trigger="final")
    assert not messages_contain(a_msgs, "secret_debater_b"), "Debater A should NOT see debater B's thinking"
    assert messages_contain(a_msgs, "secret_debater_a"), "Debater A should see own thinking"

    # Debater B does NOT see debater A's thinking, but sees own
    b_msgs, _ = build_generation_messages(state, Role.DEBATER_B, trigger="final")
    assert not messages_contain(b_msgs, "secret_debater_a"), "Debater B should NOT see debater A's thinking"
    assert messages_contain(b_msgs, "secret_debater_b"), "Debater B should see own thinking"

    print("  PASS: open_selfplay_private")


def test_open_selfplay():
    """No debater thinking (not configured); judge has private thinking."""
    print("Testing open_selfplay...")
    state = make_state_with_utterances("open_selfplay")

    # Verify think_visibility — only judge configured
    tv = state.spec.think_visibility
    assert Role.DEBATER_A not in tv, f"Debater A should not be in think_visibility, got {tv}"
    assert Role.DEBATER_B not in tv, f"Debater B should not be in think_visibility, got {tv}"
    assert tv.get(Role.JUDGE) == ThinkVisibility.PRIVATE, f"Expected PRIVATE for Judge, got {tv.get(Role.JUDGE)}"

    # Since debater thinking is DISABLED (not in think_visibility), the stripped_text
    # should be used for opponent views — "secret_*" should not appear for anyone
    # viewing opponent utterances.
    judge_msgs, _ = build_generation_messages(state, Role.JUDGE, trigger="final")
    assert not messages_contain(judge_msgs, "secret_debater_a"), "Judge should NOT see debater A thinking (disabled)"
    assert not messages_contain(judge_msgs, "secret_debater_b"), "Judge should NOT see debater B thinking (disabled)"
    # Public content should still be present
    assert messages_contain(judge_msgs, "public_debater_a"), "Judge should see debater A public content"
    assert messages_contain(judge_msgs, "public_debater_b"), "Judge should see debater B public content"

    print("  PASS: open_selfplay")


if __name__ == "__main__":
    test_open_selfplay_judgesees()
    test_open_selfplay_private()
    test_open_selfplay()
    print("\nStress test 6: ALL PASS")
    sys.exit(0)
