"""Comprehensive test suite for the debate environment.

No Tinker API needed -- all tests run offline.

Categories:
1. Schedule tests -- all protocol x round combinations, judge turns, boundaries
2. Visibility tests -- both policies, reasoning stripping, system messages
3. Reducer tests -- full playthroughs, wrong-role rejection, canonical ordering
4. Frozen state tests -- version monotonicity, immutability, fork independence
5. Runtime tests (async) -- alternation, barrier, stale ticket, CancelledError, judge
6. Env integration tests -- mock renderer + full rollout through DebateGroupBuilder
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Mapping
from unittest.mock import MagicMock

import pytest
import tinker

from tinker_cookbook.recipes.multiplayer_rl.debate.env import (
    DebateBranchGroupBuilder,
    DebateDataset,
    DebateEnv,
    DebateGroupBuilder,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.core.reducer import (
    apply_action,
    apply_judge_event,
    commit_slot_actions,
    fork_state,
    get_current_slot,
    get_eligible_roles,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.core.runtime import DebateRuntime
from tinker_cookbook.recipes.multiplayer_rl.debate.core.schedule import build_schedule
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateOutcome,
    DebateSnapshot,
    DebateSpec,
    DebateState,
    JudgeDecision,
    JudgeRequest,
    Phase,
    ProtocolKind,
    Role,
    TurnSlot,
    Utterance,
    VisibilityPolicy,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.core.visibility import (
    REGISTRY,
    _consolidate_str_messages,
    _shuffle_simultaneous,
    _strip_reasoning,
    _wrap_opponent_turn,
    all_prior,
    build_generation_messages,
    completed_rounds_only,
    get_visible_messages,
)
from tinker_cookbook.renderers import Message


# ============================================================
# Helpers
# ============================================================


def _make_spec(
    schedule: tuple[TurnSlot, ...],
    open_reasoning: bool = False,
) -> DebateSpec:
    return DebateSpec(
        debate_id="test",
        task_prompt="Which is bigger, 2 or 3?",
        answer_by_role={Role.DEBATER_A: "2", Role.DEBATER_B: "3"},
        schedule=schedule,
        open_reasoning=open_reasoning,
    )


def _make_state(
    kind: ProtocolKind = ProtocolKind.SEQUENTIAL,
    num_rounds: int = 2,
    open_reasoning: bool = False,
    include_judge_turns: bool = False,
) -> DebateState:
    schedule = build_schedule(kind, num_rounds, include_judge_turns=include_judge_turns)
    spec = _make_spec(schedule, open_reasoning=open_reasoning)
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


def _utt(role: Role, round_index: int, text: str, slot_id: int = 0) -> Utterance:
    return Utterance(
        role=role,
        round_index=round_index,
        phase=Phase.PROPOSE,
        text=text,
        token_count=len(text.split()),
        slot_id=slot_id,
    )


class MockRenderer:
    """Minimal renderer for integration tests.

    Token representation: each character is its own token (ord value).
    Enough to drive DebateEnv through a full debate without a real tokenizer.
    """

    def get_stop_sequences(self) -> list[str]:
        return ["<stop>"]

    def build_generation_prompt(self, messages: list[Message], prefill: str | None = None) -> tinker.ModelInput:
        text = "".join(m.get("content", "") or "" for m in messages)
        if prefill:
            text += prefill
        tokens = [ord(c) for c in text] if text else []
        return tinker.ModelInput.from_ints(tokens)

    def parse_response(self, tokens: list[int]) -> tuple[Message, bool]:
        text = "".join(chr(t) for t in tokens)
        return Message(role="assistant", content=text), True


class MockJudge:
    """Mock JudgeCallback for testing."""

    def __init__(self) -> None:
        self.boundary_calls: list[str] = []
        self.final_calls: list[str] = []

    async def on_boundary(self, request: JudgeRequest) -> JudgeDecision | None:
        self.boundary_calls.append(request.trigger)
        return JudgeDecision(
            round_index=request.state.rounds_completed - 1,
            verdict="A is better",
            score_delta_by_role={Role.DEBATER_A: 1.0, Role.DEBATER_B: -1.0},
        )

    async def on_final(self, request: JudgeRequest) -> DebateOutcome:
        self.final_calls.append(request.trigger)
        return DebateOutcome(
            winner=Role.DEBATER_A,
            scores_by_role={Role.DEBATER_A: 1.0, Role.DEBATER_B: 0.0},
        )


# ============================================================
# 1. Schedule tests
# ============================================================


class TestSchedule:
    def test_invalid_rounds(self):
        with pytest.raises(ValueError, match="num_rounds must be >= 1"):
            build_schedule(ProtocolKind.SEQUENTIAL, 0)

    @pytest.mark.parametrize("num_rounds", [1, 2, 3])
    def test_sequential_structure(self, num_rounds: int):
        slots = build_schedule(ProtocolKind.SEQUENTIAL, num_rounds)
        assert len(slots) == num_rounds * 2
        for i, slot in enumerate(slots):
            assert slot.slot_id == i
            assert slot.visibility_policy == VisibilityPolicy.ALL_PRIOR
            assert slot.round_index == i // 2
            if i % 2 == 0:
                assert slot.actors == (Role.DEBATER_A,)
                assert not slot.boundary_after
            else:
                assert slot.actors == (Role.DEBATER_B,)
                assert slot.boundary_after

    @pytest.mark.parametrize("num_rounds", [1, 2, 3])
    def test_simultaneous_structure(self, num_rounds: int):
        slots = build_schedule(ProtocolKind.SIMULTANEOUS, num_rounds)
        assert len(slots) == num_rounds
        for i, slot in enumerate(slots):
            assert slot.slot_id == i
            assert slot.actors == (Role.DEBATER_A, Role.DEBATER_B)
            assert slot.visibility_policy == VisibilityPolicy.COMPLETED_ROUNDS_ONLY
            assert slot.boundary_after

    @pytest.mark.parametrize("num_rounds", [2, 3])
    def test_hybrid_structure(self, num_rounds: int):
        slots = build_schedule(ProtocolKind.HYBRID, num_rounds)
        expected_len = 1 + (num_rounds - 1) * 2
        assert len(slots) == expected_len
        # Round 0: simultaneous
        assert slots[0].actors == (Role.DEBATER_A, Role.DEBATER_B)
        assert slots[0].visibility_policy == VisibilityPolicy.COMPLETED_ROUNDS_ONLY
        assert slots[0].boundary_after
        assert slots[0].phase == Phase.PROPOSE
        # Subsequent: sequential critiques
        for s in slots[1:]:
            assert len(s.actors) == 1
            assert s.visibility_policy == VisibilityPolicy.ALL_PRIOR
            assert s.phase == Phase.CRITIQUE

    def test_sequential_phases(self):
        slots = build_schedule(ProtocolKind.SEQUENTIAL, 2)
        assert slots[0].phase == Phase.PROPOSE
        assert slots[1].phase == Phase.PROPOSE
        assert slots[2].phase == Phase.CRITIQUE
        assert slots[3].phase == Phase.CRITIQUE

    @pytest.mark.parametrize("kind", list(ProtocolKind))
    def test_judge_turns_appended(self, kind: ProtocolKind):
        no_judge = build_schedule(kind, 2)
        with_judge = build_schedule(kind, 2, include_judge_turns=True)
        assert len(with_judge) == len(no_judge) + 2 * 2

    def test_judge_turn_phases(self):
        slots = build_schedule(ProtocolKind.SEQUENTIAL, 1, include_judge_turns=True)
        assert len(slots) == 4
        assert slots[2].phase == Phase.JUDGE_QUERY
        assert slots[2].actors == (Role.JUDGE,)
        assert slots[3].phase == Phase.JUDGE_VERDICT
        assert slots[3].actors == (Role.JUDGE,)

    @pytest.mark.parametrize("kind", list(ProtocolKind))
    @pytest.mark.parametrize("num_rounds", [1, 2, 3])
    def test_slot_ids_sequential(self, kind: ProtocolKind, num_rounds: int):
        slots = build_schedule(kind, num_rounds)
        assert [s.slot_id for s in slots] == list(range(len(slots)))

    @pytest.mark.parametrize("kind", list(ProtocolKind))
    @pytest.mark.parametrize("num_rounds", [1, 2, 3])
    def test_boundary_count_equals_rounds(self, kind: ProtocolKind, num_rounds: int):
        slots = build_schedule(kind, num_rounds)
        assert sum(1 for s in slots if s.boundary_after) == num_rounds

    @pytest.mark.parametrize("kind", list(ProtocolKind))
    @pytest.mark.parametrize("num_rounds", [1, 2, 3])
    def test_round_indices_valid(self, kind: ProtocolKind, num_rounds: int):
        slots = build_schedule(kind, num_rounds)
        for s in slots:
            assert 0 <= s.round_index < num_rounds


# ============================================================
# 2. Visibility tests
# ============================================================


class TestVisibility:
    def test_registry_populated(self):
        assert VisibilityPolicy.ALL_PRIOR in REGISTRY
        assert VisibilityPolicy.COMPLETED_ROUNDS_ONLY in REGISTRY

    def test_strip_reasoning(self):
        assert _strip_reasoning("A <thinking>secret</thinking> B") == "A  B"
        assert _strip_reasoning("no tags") == "no tags"
        assert _strip_reasoning("<thinking>\nline\n</thinking> after") == "after"
        # Short form <think> tags
        assert _strip_reasoning("before <think>secret</think> after") == "before  after"
        assert _strip_reasoning("<think>line</think> tail") == "tail"

    def test_all_prior_returns_utterances(self):
        schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1)
        spec = _make_spec(schedule)
        transcript = (_utt(Role.DEBATER_A, 0, "hello"), _utt(Role.DEBATER_B, 0, "hi"))
        state = replace(_make_state(), spec=spec, transcript=transcript, slot_index=2, rounds_completed=1)
        # V2: all_prior returns list[Utterance]
        utts = all_prior(state, Role.DEBATER_A)
        assert len(utts) == 2
        assert utts[0].role == Role.DEBATER_A
        assert utts[1].role == Role.DEBATER_B

    def test_reasoning_strip_in_visible_messages(self):
        schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1)
        spec = _make_spec(schedule, open_reasoning=False)
        transcript = (
            _utt(Role.DEBATER_A, 0, "A <thinking>secret</thinking> visible"),
            _utt(Role.DEBATER_B, 0, "B <thinking>secret</thinking> visible"),
        )
        state = replace(
            _make_state(), spec=spec, transcript=transcript,
            slot_index=2, rounds_completed=1,
        )
        # V2: reasoning stripping happens at message conversion (get_visible_messages)
        msgs = get_visible_messages(state, Role.DEBATER_A)
        # Own turn (assistant) keeps reasoning
        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        assert any("<thinking>" in m["content"] for m in assistant_msgs)
        # Opponent turn (user with <opponent_turn>) has reasoning stripped
        user_msgs = [m for m in msgs if m["role"] == "user" and "opponent_turn" in m.get("content", "")]
        assert all("<thinking>" not in m["content"] for m in user_msgs)

    def test_reasoning_kept_open(self):
        schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1)
        spec = _make_spec(schedule, open_reasoning=True)
        transcript = (
            _utt(Role.DEBATER_A, 0, "A <thinking>visible</thinking>"),
            _utt(Role.DEBATER_B, 0, "B <thinking>visible</thinking>"),
        )
        state = replace(
            _make_state(open_reasoning=True), spec=spec, transcript=transcript,
            slot_index=2, rounds_completed=1,
        )
        msgs = get_visible_messages(state, Role.DEBATER_A)
        # With open_reasoning, opponent text preserves reasoning
        user_msgs = [m for m in msgs if m["role"] == "user" and "opponent_turn" in m.get("content", "")]
        assert any("<thinking>" in m["content"] for m in user_msgs)

    def test_completed_rounds_only_filters(self):
        schedule = build_schedule(ProtocolKind.SIMULTANEOUS, 2)
        spec = _make_spec(schedule)
        transcript = (
            _utt(Role.DEBATER_A, 0, "r0 A"),
            _utt(Role.DEBATER_B, 0, "r0 B"),
            _utt(Role.DEBATER_A, 1, "r1 A"),
        )
        state = replace(
            _make_state(kind=ProtocolKind.SIMULTANEOUS), spec=spec,
            transcript=transcript, slot_index=1, rounds_completed=1,
        )
        msgs = completed_rounds_only(state, Role.DEBATER_A)
        assert len(msgs) == 2  # only round 0

    def test_system_message_debater(self):
        state = _make_state()
        msgs = get_visible_messages(state, Role.DEBATER_A)
        # V2: system is persona-only, question is separate message
        assert msgs[0]["role"] == "system"
        assert "debater_a" in msgs[0]["content"]
        assert "Which is bigger" not in msgs[0]["content"]  # task_prompt in question, not system
        # Question message
        assert msgs[1]["role"] == "user"
        assert "Which is bigger" in msgs[1]["content"]
        assert "Your assigned position: 2" in msgs[1]["content"]

    def test_system_message_judge(self):
        state = _make_state()
        msgs = get_visible_messages(state, Role.JUDGE)
        assert msgs[0]["role"] == "system"
        assert "judge" in msgs[0]["content"]
        assert "Evaluate" in msgs[0]["content"]
        assert "Defend answer" not in msgs[0]["content"]

    def test_schedule_exhausted_fallback(self):
        schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1)
        spec = _make_spec(schedule)
        transcript = (_utt(Role.DEBATER_A, 0, "A"), _utt(Role.DEBATER_B, 0, "B"))
        state = replace(
            _make_state(), spec=spec, transcript=transcript,
            slot_index=len(schedule), rounds_completed=1,
        )
        msgs = get_visible_messages(state, Role.DEBATER_A)
        # V2: system + question + 2 utterances = 4
        assert len(msgs) == 4


# ============================================================
# 2b. V2 visibility helper tests
# ============================================================


class TestWrapOpponentTurn:
    def test_format_and_label(self):
        utt = _utt(Role.DEBATER_A, 0, "my argument")
        result = _wrap_opponent_turn(utt, open_reasoning=True)
        assert '<opponent_turn agent="Debater A" phase="propose">' in result
        assert "my argument" in result
        assert "</opponent_turn>" in result

    def test_label_debater_b(self):
        utt = _utt(Role.DEBATER_B, 0, "counter")
        result = _wrap_opponent_turn(utt, open_reasoning=True)
        assert 'agent="Debater B"' in result

    def test_reasoning_stripped_when_closed(self):
        utt = _utt(Role.DEBATER_A, 0, "visible <thinking>secret</thinking> end")
        result = _wrap_opponent_turn(utt, open_reasoning=False)
        assert "<thinking>" not in result
        assert "visible" in result
        assert "end" in result

    def test_reasoning_preserved_when_open(self):
        utt = _utt(Role.DEBATER_A, 0, "visible <thinking>kept</thinking> end")
        result = _wrap_opponent_turn(utt, open_reasoning=True)
        assert "<thinking>kept</thinking>" in result

    def test_think_short_form_stripped(self):
        utt = _utt(Role.DEBATER_A, 0, "before <think>hidden</think> after")
        result = _wrap_opponent_turn(utt, open_reasoning=False)
        assert "<think>" not in result
        assert "before" in result


class TestShuffleSimultaneous:
    def _sim_state(self, transcript: tuple[Utterance, ...], debate_id: str = "test") -> DebateState:
        schedule = build_schedule(ProtocolKind.SIMULTANEOUS, 1)
        spec = DebateSpec(
            debate_id=debate_id,
            task_prompt="Q",
            answer_by_role={Role.DEBATER_A: "A", Role.DEBATER_B: "B"},
            schedule=schedule,
            open_reasoning=False,
        )
        return DebateState(
            spec=spec, slot_index=1, rounds_completed=1,
            transcript=transcript, pending_simultaneous={},
            judge_trace=(), done=True, outcome=None,
        )

    def test_no_op_for_debaters(self):
        transcript = (
            _utt(Role.DEBATER_A, 0, "A", slot_id=0),
            _utt(Role.DEBATER_B, 0, "B", slot_id=0),
        )
        state = self._sim_state(transcript)
        result = _shuffle_simultaneous(list(transcript), Role.DEBATER_A, state)
        assert [u.role for u in result] == [Role.DEBATER_A, Role.DEBATER_B]

    def test_shuffles_for_judge(self):
        transcript = (
            _utt(Role.DEBATER_A, 0, "A", slot_id=0),
            _utt(Role.DEBATER_B, 0, "B", slot_id=0),
        )
        state = self._sim_state(transcript)
        result = _shuffle_simultaneous(list(transcript), Role.JUDGE, state)
        # Should still have both utterances
        assert len(result) == 2
        assert set(u.role for u in result) == {Role.DEBATER_A, Role.DEBATER_B}

    def test_deterministic(self):
        transcript = (
            _utt(Role.DEBATER_A, 0, "A", slot_id=0),
            _utt(Role.DEBATER_B, 0, "B", slot_id=0),
        )
        state = self._sim_state(transcript)
        r1 = _shuffle_simultaneous(list(transcript), Role.JUDGE, state)
        r2 = _shuffle_simultaneous(list(transcript), Role.JUDGE, state)
        assert [u.role for u in r1] == [u.role for u in r2]

    def test_different_debate_ids_can_differ(self):
        """Different debate_ids produce different seeds; at least one should differ from canonical."""
        transcript = (
            _utt(Role.DEBATER_A, 0, "A", slot_id=0),
            _utt(Role.DEBATER_B, 0, "B", slot_id=0),
        )
        orders = set()
        for i in range(20):
            state = self._sim_state(transcript, debate_id=f"debate_{i}")
            result = _shuffle_simultaneous(list(transcript), Role.JUDGE, state)
            orders.add(tuple(u.role for u in result))
        # With 20 different seeds, should see both orderings
        assert len(orders) == 2

    def test_sequential_slots_not_shuffled(self):
        """Single-actor slots are not shuffled even for judge."""
        schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1)
        spec = DebateSpec(
            debate_id="test",
            task_prompt="Q",
            answer_by_role={Role.DEBATER_A: "A", Role.DEBATER_B: "B"},
            schedule=schedule,
            open_reasoning=False,
        )
        transcript = (
            _utt(Role.DEBATER_A, 0, "A", slot_id=0),
            _utt(Role.DEBATER_B, 0, "B", slot_id=1),
        )
        state = DebateState(
            spec=spec, slot_index=2, rounds_completed=1,
            transcript=transcript, pending_simultaneous={},
            judge_trace=(), done=True, outcome=None,
        )
        result = _shuffle_simultaneous(list(transcript), Role.JUDGE, state)
        assert [u.role for u in result] == [Role.DEBATER_A, Role.DEBATER_B]


class TestConsolidateMessages:
    def test_empty(self):
        assert _consolidate_str_messages([]) == []

    def test_no_merge_different_roles(self):
        msgs = [
            Message(role="user", content="a"),
            Message(role="assistant", content="b"),
            Message(role="user", content="c"),
        ]
        assert len(_consolidate_str_messages(msgs)) == 3

    def test_merge_adjacent_same_role(self):
        msgs = [
            Message(role="user", content="hello"),
            Message(role="user", content="world"),
        ]
        result = _consolidate_str_messages(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "hello\n\nworld"

    def test_skip_system(self):
        msgs = [
            Message(role="system", content="sys1"),
            Message(role="system", content="sys2"),
        ]
        result = _consolidate_str_messages(msgs)
        assert len(result) == 2  # system messages not merged

    def test_type_gated_non_str_content(self):
        msgs = [
            Message(role="user", content=[{"type": "text", "text": "a"}]),
            Message(role="user", content="b"),
        ]
        result = _consolidate_str_messages(msgs)
        assert len(result) == 2  # non-str content blocks merge


class TestBuildGenerationMessages:
    def test_message_sequence(self):
        schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1)
        spec = _make_spec(schedule)
        transcript = (_utt(Role.DEBATER_A, 0, "hello"),)
        state = replace(
            _make_state(), spec=spec, transcript=transcript, slot_index=1,
        )
        msgs, prefill = build_generation_messages(state, Role.DEBATER_B)
        # system(0), question(1), opponent transcript(2) = 3
        # No instruction: default.yaml debater user template is empty, no think/fields
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"  # question
        assert "Which is bigger" in msgs[1]["content"]
        assert msgs[2]["role"] == "user"  # opponent turn (wrapped)
        assert '<opponent_turn agent="Debater A"' in msgs[2]["content"]

    def test_trigger_override(self):
        state = _make_state()
        msgs, _prefill = build_generation_messages(state, Role.JUDGE, trigger="final")
        # Judge final: system + question + instruction = 3 (empty transcript)
        assert len(msgs) == 3
        # Last message is the instruction with field format instructions
        last = msgs[-1]
        assert last["role"] == "user"
        assert "<decision>" in last["content"]  # field instructions from default.yaml

    def test_returns_prefill(self):
        state = _make_state()
        msgs, prefill = build_generation_messages(state, Role.DEBATER_A)
        # Default YAML has no prefill section, so prefill should be None
        assert prefill is None

    def test_opponent_wrapped_in_xml(self):
        schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1)
        spec = _make_spec(schedule)
        transcript = (_utt(Role.DEBATER_A, 0, "A arg"), _utt(Role.DEBATER_B, 0, "B arg"))
        state = replace(
            _make_state(), spec=spec, transcript=transcript,
            slot_index=2, rounds_completed=1,
        )
        msgs, _ = build_generation_messages(state, Role.DEBATER_A)
        opponent_msgs = [m for m in msgs if m["role"] == "user" and "opponent_turn" in str(m.get("content", ""))]
        assert len(opponent_msgs) == 1  # exactly one opponent turn (B)
        assert 'agent="Debater B"' in opponent_msgs[0]["content"]
        assert "</opponent_turn>" in opponent_msgs[0]["content"]


# ============================================================
# 3. Reducer tests
# ============================================================


class TestReducer:
    def test_sequential_full_playthrough(self):
        state = _make_state(kind=ProtocolKind.SEQUENTIAL, num_rounds=2)
        expectations = [
            (Role.DEBATER_A, False, False),
            (Role.DEBATER_B, True, False),
            (Role.DEBATER_A, False, False),
            (Role.DEBATER_B, True, True),
        ]
        for i, (role, boundary, done) in enumerate(expectations):
            result = apply_action(state, role, f"turn {i}", 2)
            assert result.boundary_reached == boundary
            assert result.episode_done == done
            state = result.new_state
        assert state.done and len(state.transcript) == 4 and state.rounds_completed == 2

    def test_simultaneous_full_playthrough(self):
        state = _make_state(kind=ProtocolKind.SIMULTANEOUS, num_rounds=2)
        for rnd in range(2):
            r = apply_action(state, Role.DEBATER_A, f"A r{rnd}", 2)
            assert len(r.committed) == 0  # buffered
            state = r.new_state
            r = apply_action(state, Role.DEBATER_B, f"B r{rnd}", 2)
            assert len(r.committed) == 2
            assert r.committed[0].role == Role.DEBATER_A
            assert r.committed[1].role == Role.DEBATER_B
            state = r.new_state
        assert state.done and len(state.transcript) == 4

    def test_hybrid_full_playthrough(self):
        state = _make_state(kind=ProtocolKind.HYBRID, num_rounds=2)
        r = apply_action(state, Role.DEBATER_A, "A", 1)
        state = r.new_state
        r = apply_action(state, Role.DEBATER_B, "B", 1)
        state = r.new_state
        assert state.rounds_completed == 1
        r = apply_action(state, Role.DEBATER_A, "A", 1)
        state = r.new_state
        r = apply_action(state, Role.DEBATER_B, "B", 1)
        state = r.new_state
        assert state.done and len(state.transcript) == 4

    def test_simultaneous_canonical_order_reversed(self):
        """B submits first, canonical order is still A, B."""
        state = _make_state(kind=ProtocolKind.SIMULTANEOUS, num_rounds=1)
        r = apply_action(state, Role.DEBATER_B, "B first", 2)
        r = apply_action(r.new_state, Role.DEBATER_A, "A second", 2)
        assert r.committed[0].role == Role.DEBATER_A
        assert r.committed[1].role == Role.DEBATER_B

    def test_wrong_role_rejected(self):
        state = _make_state()
        with pytest.raises(ValueError, match="not eligible"):
            apply_action(state, Role.DEBATER_B, "wrong", 1)

    def test_simultaneous_duplicate_rejected(self):
        state = _make_state(kind=ProtocolKind.SIMULTANEOUS)
        r = apply_action(state, Role.DEBATER_A, "A", 1)
        with pytest.raises(ValueError, match="not eligible"):
            apply_action(r.new_state, Role.DEBATER_A, "A again", 1)

    def test_commit_slot_actions(self):
        state = _make_state(kind=ProtocolKind.SIMULTANEOUS)
        result = commit_slot_actions(state, {
            Role.DEBATER_B: ("B", 1), Role.DEBATER_A: ("A", 1),
        })
        assert len(result.committed) == 2
        assert result.committed[0].role == Role.DEBATER_A

    def test_commit_slot_actions_wrong_roles(self):
        state = _make_state(kind=ProtocolKind.SEQUENTIAL)
        with pytest.raises(ValueError, match="Expected actions"):
            commit_slot_actions(state, {
                Role.DEBATER_A: ("A", 1), Role.DEBATER_B: ("B", 1),
            })

    def test_apply_judge_event(self):
        state = _make_state()
        d = JudgeDecision(round_index=0, verdict="ok", score_delta_by_role={})
        new = apply_judge_event(state, d)
        assert len(new.judge_trace) == 1 and new.judge_trace[0] is d

    def test_get_current_slot(self):
        state = _make_state()
        slot = get_current_slot(state)
        assert slot is not None and slot.slot_id == 0

    def test_get_current_slot_exhausted(self):
        state = replace(_make_state(num_rounds=1), slot_index=2)
        assert get_current_slot(state) is None

    def test_get_eligible_roles_sequential(self):
        assert get_eligible_roles(_make_state()) == frozenset({Role.DEBATER_A})

    def test_get_eligible_roles_simultaneous(self):
        assert get_eligible_roles(_make_state(kind=ProtocolKind.SIMULTANEOUS)) == frozenset(
            {Role.DEBATER_A, Role.DEBATER_B}
        )

    def test_get_eligible_roles_done(self):
        assert get_eligible_roles(replace(_make_state(), done=True)) == frozenset()


# ============================================================
# 4. Frozen state tests
# ============================================================


class TestFrozenState:
    def test_state_immutable(self):
        state = _make_state()
        with pytest.raises(AttributeError):
            state.done = True  # type: ignore[misc]
        with pytest.raises(AttributeError):
            state.slot_index = 5  # type: ignore[misc]

    def test_utterance_immutable(self):
        u = _utt(Role.DEBATER_A, 0, "hi")
        with pytest.raises(AttributeError):
            u.text = "changed"  # type: ignore[misc]

    def test_spec_immutable(self):
        spec = _make_spec(build_schedule(ProtocolKind.SEQUENTIAL, 1))
        with pytest.raises(AttributeError):
            spec.debate_id = "changed"  # type: ignore[misc]

    def test_turn_slot_hashable(self):
        slot = TurnSlot(
            slot_id=0, round_index=0, phase=Phase.PROPOSE,
            actors=(Role.DEBATER_A,), boundary_after=False,
            visibility_policy=VisibilityPolicy.ALL_PRIOR,
        )
        hash(slot)
        {slot}  # can go in a set

    def test_version_monotonic_sequential(self):
        state = _make_state(kind=ProtocolKind.SEQUENTIAL, num_rounds=2)
        versions = [state.version]
        for role in [Role.DEBATER_A, Role.DEBATER_B, Role.DEBATER_A, Role.DEBATER_B]:
            r = apply_action(state, role, "x", 1)
            state = r.new_state
            versions.append(state.version)
        for i in range(1, len(versions)):
            assert versions[i] > versions[i - 1]

    def test_version_increases_during_buffer(self):
        state = _make_state(kind=ProtocolKind.SIMULTANEOUS)
        v0 = state.version
        r = apply_action(state, Role.DEBATER_A, "x", 1)
        assert r.new_state.version > v0

    def test_fork_returns_same(self):
        state = _make_state()
        assert fork_state(state) is state


# ============================================================
# 5. Runtime tests (async)
# ============================================================


class TestRuntime:
    @pytest.mark.asyncio
    async def test_sequential_alternation(self):
        runtime = DebateRuntime(_make_state(kind=ProtocolKind.SEQUENTIAL, num_rounds=2))
        turns: list[Role] = []

        async def play(role: Role):
            while True:
                ticket = await runtime.wait_for_turn(role)
                if ticket is None:
                    break
                result = await runtime.submit(ticket, f"{role.value}", 1)
                turns.append(role)
                if result.episode_done:
                    break

        await asyncio.gather(play(Role.DEBATER_A), play(Role.DEBATER_B))
        assert turns == [Role.DEBATER_A, Role.DEBATER_B, Role.DEBATER_A, Role.DEBATER_B]
        assert runtime.state.done

    @pytest.mark.asyncio
    async def test_simultaneous_barrier(self):
        runtime = DebateRuntime(_make_state(kind=ProtocolKind.SIMULTANEOUS, num_rounds=1))

        async def play(role: Role):
            ticket = await runtime.wait_for_turn(role)
            assert ticket is not None
            return await runtime.submit(ticket, f"{role.value}", 1)

        results = await asyncio.gather(play(Role.DEBATER_A), play(Role.DEBATER_B))
        assert all(r.episode_done for r in results)
        assert len(runtime.state.transcript) == 2

    @pytest.mark.asyncio
    async def test_simultaneous_multi_round(self):
        runtime = DebateRuntime(_make_state(kind=ProtocolKind.SIMULTANEOUS, num_rounds=2))
        counts = {Role.DEBATER_A: 0, Role.DEBATER_B: 0}

        async def play(role: Role):
            while True:
                t = await runtime.wait_for_turn(role)
                if t is None:
                    break
                r = await runtime.submit(t, f"{role.value}", 1)
                counts[role] += 1
                if r.episode_done:
                    break

        await asyncio.gather(play(Role.DEBATER_A), play(Role.DEBATER_B))
        assert counts == {Role.DEBATER_A: 2, Role.DEBATER_B: 2}
        assert runtime.state.done

    @pytest.mark.asyncio
    async def test_hybrid_integration(self):
        runtime = DebateRuntime(_make_state(kind=ProtocolKind.HYBRID, num_rounds=2))

        async def play(role: Role):
            while True:
                t = await runtime.wait_for_turn(role)
                if t is None:
                    break
                r = await runtime.submit(t, f"{role.value}", 1)
                if r.episode_done:
                    break

        await asyncio.gather(play(Role.DEBATER_A), play(Role.DEBATER_B))
        assert runtime.state.done and len(runtime.state.transcript) == 4

    @pytest.mark.asyncio
    async def test_stale_ticket(self):
        runtime = DebateRuntime(_make_state(kind=ProtocolKind.SEQUENTIAL, num_rounds=1))
        ticket = await runtime.wait_for_turn(Role.DEBATER_A)
        assert ticket is not None
        await runtime.submit(ticket, "A", 1)
        with pytest.raises(ValueError, match="Stale ticket"):
            await runtime.submit(ticket, "A again", 1)

    @pytest.mark.asyncio
    async def test_cancelled_error_safety(self):
        runtime = DebateRuntime(_make_state(kind=ProtocolKind.SEQUENTIAL, num_rounds=1))
        task = asyncio.create_task(runtime.wait_for_turn(Role.DEBATER_B))
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        ticket = await runtime.wait_for_turn(Role.DEBATER_A)
        assert ticket is not None

    @pytest.mark.asyncio
    async def test_judge_callback_timing(self):
        judge = MockJudge()
        runtime = DebateRuntime(
            _make_state(kind=ProtocolKind.SEQUENTIAL, num_rounds=1),
            judge_callback=judge,
        )

        async def play(role: Role):
            while True:
                ticket = await runtime.wait_for_turn(role)
                if ticket is None:
                    break
                result = await runtime.submit(ticket, f"{role.value}", 1)
                if result.episode_done:
                    break

        await asyncio.gather(play(Role.DEBATER_A), play(Role.DEBATER_B))
        assert len(judge.boundary_calls) == 1
        assert len(judge.final_calls) == 1
        assert runtime.state.outcome is not None
        assert runtime.state.outcome.winner == Role.DEBATER_A
        assert len(runtime.state.judge_trace) == 1

    @pytest.mark.asyncio
    async def test_step_reward(self):
        fn = lambda b, a, role, utt: 1.0 if role == Role.DEBATER_A else -1.0
        runtime = DebateRuntime(
            _make_state(kind=ProtocolKind.SEQUENTIAL, num_rounds=1),
            step_reward_fn=fn,
        )
        ticket = await runtime.wait_for_turn(Role.DEBATER_A)
        assert ticket is not None
        result = await runtime.submit(ticket, "A", 1)
        assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_simultaneous_step_reward(self):
        """Both barrier arrivers get their correct utterance for reward."""
        rewards: dict[Role, float] = {}

        def fn(b, a, role, utt):
            r = float(len(utt.text)) if utt else -1.0
            rewards[role] = r
            return r

        runtime = DebateRuntime(
            _make_state(kind=ProtocolKind.SIMULTANEOUS, num_rounds=1),
            step_reward_fn=fn,
        )

        async def play(role: Role):
            t = await runtime.wait_for_turn(role)
            if t:
                await runtime.submit(t, f"{role.value}_text", 5)

        await asyncio.gather(play(Role.DEBATER_A), play(Role.DEBATER_B))
        assert rewards[Role.DEBATER_A] == len("debater_a_text")
        assert rewards[Role.DEBATER_B] == len("debater_b_text")

    @pytest.mark.asyncio
    async def test_obs_consistent(self):
        runtime = DebateRuntime(_make_state(kind=ProtocolKind.SEQUENTIAL, num_rounds=1))
        ticket = await runtime.wait_for_turn(Role.DEBATER_A)
        assert ticket is not None
        result = await runtime.submit(ticket, "A", 1)
        # V2: cache removed, but messages should be equal
        assert result.messages == runtime._get_messages(runtime.state, Role.DEBATER_A)

    @pytest.mark.asyncio
    async def test_wait_returns_none_when_done(self):
        runtime = DebateRuntime(_make_state(kind=ProtocolKind.SEQUENTIAL, num_rounds=1))
        t = await runtime.wait_for_turn(Role.DEBATER_A)
        await runtime.submit(t, "A", 1)
        t = await runtime.wait_for_turn(Role.DEBATER_B)
        await runtime.submit(t, "B", 1)
        assert await runtime.wait_for_turn(Role.DEBATER_A) is None
        assert await runtime.wait_for_turn(Role.DEBATER_B) is None

    def test_snapshot(self):
        runtime = DebateRuntime(_make_state())
        snap = runtime.snapshot("llama3", ProtocolKind.SEQUENTIAL, {"num_rounds": 2})
        assert snap.state is runtime.state
        assert snap.renderer_name == "llama3"


# ============================================================
# 6. Env integration tests
# ============================================================


class TestEnvIntegration:
    """Tests using MockRenderer to drive full debates through DebateEnv."""

    @pytest.mark.asyncio
    async def test_sequential_full_rollout(self):
        """Full sequential debate: DebateGroupBuilder -> DebateEnv -> rollout."""
        renderer = MockRenderer()
        builder = DebateGroupBuilder(
            task_prompt="Is 2 > 3?", answer_a="yes", answer_b="no",
            renderer=renderer, protocol_kind=ProtocolKind.SEQUENTIAL, num_rounds=1,
        )
        envs = await builder.make_envs()
        assert len(envs) == 2

        async def rollout(env: DebateEnv) -> list:
            await env.initial_observation()
            results = []
            tokens = [ord(c) for c in f"I am {env.role.value}"]
            result = await env.step(tokens)
            results.append(result)
            while not result.episode_done:
                tokens = [ord(c) for c in f"{env.role.value} continues"]
                result = await env.step(tokens)
                results.append(result)
            return results

        all_results = await asyncio.gather(*[rollout(e) for e in envs])  # type: ignore[arg-type]
        for agent_results in all_results:
            assert agent_results[-1].episode_done
        assert builder._runtime.state.done
        assert len(builder._runtime.state.transcript) == 2

    @pytest.mark.asyncio
    async def test_simultaneous_full_rollout(self):
        renderer = MockRenderer()
        builder = DebateGroupBuilder(
            task_prompt="Q", answer_a="A", answer_b="B",
            renderer=renderer, protocol_kind=ProtocolKind.SIMULTANEOUS, num_rounds=2,
        )
        envs = await builder.make_envs()

        async def rollout(env: DebateEnv) -> list:
            await env.initial_observation()
            results = []
            while True:
                tokens = [ord(c) for c in f"{env.role.value}"]
                result = await env.step(tokens)
                results.append(result)
                if result.episode_done:
                    break
            return results

        all_results = await asyncio.gather(*[rollout(e) for e in envs])  # type: ignore[arg-type]
        for agent_results in all_results:
            assert agent_results[-1].episode_done
        assert builder._runtime.state.done
        assert len(builder._runtime.state.transcript) == 4

    @pytest.mark.asyncio
    async def test_hybrid_full_rollout(self):
        renderer = MockRenderer()
        builder = DebateGroupBuilder(
            task_prompt="Q", answer_a="A", answer_b="B",
            renderer=renderer, protocol_kind=ProtocolKind.HYBRID, num_rounds=2,
        )
        envs = await builder.make_envs()

        async def rollout(env: DebateEnv) -> list:
            await env.initial_observation()
            results = []
            while True:
                tokens = [ord(c) for c in f"{env.role.value}"]
                result = await env.step(tokens)
                results.append(result)
                if result.episode_done:
                    break
            return results

        all_results = await asyncio.gather(*[rollout(e) for e in envs])  # type: ignore[arg-type]
        for agent_results in all_results:
            assert agent_results[-1].episode_done
        assert builder._runtime.state.done
        assert len(builder._runtime.state.transcript) == 4

    @pytest.mark.asyncio
    async def test_group_rewards_with_judge(self):
        """Full rollout with judge callback, then compute_group_rewards."""

        def outcome_fn(outcome: DebateOutcome) -> Mapping[Role, float]:
            return outcome.scores_by_role

        judge = MockJudge()
        renderer = MockRenderer()
        builder = DebateGroupBuilder(
            task_prompt="Q", answer_a="A", answer_b="B",
            renderer=renderer, protocol_kind=ProtocolKind.SEQUENTIAL, num_rounds=1,
            judge_callback=judge, outcome_reward_fn=outcome_fn,
        )
        envs = await builder.make_envs()

        async def rollout(env: DebateEnv):
            await env.initial_observation()
            while True:
                tokens = [ord(c) for c in f"{env.role.value}"]
                result = await env.step(tokens)
                if result.episode_done:
                    break

        await asyncio.gather(*[rollout(e) for e in envs])  # type: ignore[arg-type]
        fake_trajs = [MagicMock() for _ in envs]
        rewards = await builder.compute_group_rewards(fake_trajs, envs)
        for env, (reward, _) in zip(envs, rewards):
            assert isinstance(env, DebateEnv)
            expected = 1.0 if env.role == Role.DEBATER_A else 0.0
            assert reward == expected

    @pytest.mark.asyncio
    async def test_branch_builder_independent(self):
        """Branch from mid-debate snapshot creates independent runtime."""
        renderer = MockRenderer()
        builder = DebateGroupBuilder(
            task_prompt="Q", answer_a="A", answer_b="B",
            renderer=renderer, protocol_kind=ProtocolKind.SEQUENTIAL, num_rounds=2,
        )
        await builder.make_envs()

        # Play round 0 via runtime.
        t = await builder._runtime.wait_for_turn(Role.DEBATER_A)
        await builder._runtime.submit(t, "A round 0", 3)
        t = await builder._runtime.wait_for_turn(Role.DEBATER_B)
        await builder._runtime.submit(t, "B round 0", 3)
        assert builder._runtime.state.rounds_completed == 1

        # Snapshot and branch.
        snap = builder._runtime.snapshot("mock", ProtocolKind.SEQUENTIAL, {})
        branch = DebateBranchGroupBuilder(snapshot=snap, renderer=renderer)
        await branch.make_envs()

        # Advance original.
        t = await builder._runtime.wait_for_turn(Role.DEBATER_A)
        await builder._runtime.submit(t, "A round 1 original", 4)

        # Branch should be independent.
        assert branch._runtime.state.rounds_completed == 1
        assert len(branch._runtime.state.transcript) == 2

    def test_env_count_default(self):
        builder = DebateGroupBuilder(
            task_prompt="test", answer_a="A", answer_b="B",
            renderer=MagicMock(), protocol_kind=ProtocolKind.SEQUENTIAL, num_rounds=1,
        )
        envs = asyncio.get_event_loop().run_until_complete(builder.make_envs())
        assert len(envs) == 2 and all(isinstance(e, DebateEnv) for e in envs)

    def test_partial_roles_rejected(self):
        builder = DebateGroupBuilder(
            task_prompt="test", answer_a="A", answer_b="B",
            renderer=MagicMock(), protocol_kind=ProtocolKind.SEQUENTIAL,
            num_rounds=1, include_roles=(Role.DEBATER_A,),
        )
        with pytest.raises(ValueError, match="Schedule requires roles"):
            asyncio.get_event_loop().run_until_complete(builder.make_envs())

    def test_dataset_batching(self):
        problems = [("q1", "a1", "b1"), ("q2", "a2", "b2"), ("q3", "a3", "b3")]
        ds = DebateDataset(
            problems=problems, batch_size=2, renderer=MagicMock(),
            protocol_kind=ProtocolKind.SEQUENTIAL, num_rounds=1,
        )
        assert len(ds) == 2
        batch = ds.get_batch(0)
        assert len(batch) == 2
        assert batch[0].task_prompt == "q1"

    def test_logging_tags(self):
        for kind in ProtocolKind:
            builder = DebateGroupBuilder(
                task_prompt="Q", answer_a="A", answer_b="B",
                renderer=MagicMock(), protocol_kind=kind, num_rounds=1,
            )
            assert builder.logging_tags() == ["debate", kind.value]

    def test_compute_rewards_no_outcome_fn(self):
        builder = DebateGroupBuilder(
            task_prompt="test", answer_a="A", answer_b="B",
            renderer=MagicMock(), protocol_kind=ProtocolKind.SEQUENTIAL, num_rounds=1,
        )
        envs = asyncio.get_event_loop().run_until_complete(builder.make_envs())
        fake_trajs = [MagicMock() for _ in envs]
        rewards = asyncio.get_event_loop().run_until_complete(
            builder.compute_group_rewards(fake_trajs, envs)
        )
        assert all(r == (0.0, {}) for r in rewards)
