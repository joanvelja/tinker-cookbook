"""Tests for visibility policies (V2 message assembly)."""

from __future__ import annotations


from tinker_cookbook.recipes.multiplayer_rl.debate.core.schedule import build_schedule
from tinker_cookbook.recipes.multiplayer_rl.debate.tests.conftest import make_spec
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateProblemSpec,
    DebateSpec,
    DebateState,
    Phase,
    ProtocolKind,
    Role,
    ScoringMode,
    ThinkVisibility,
    TurnSlot,
    Utterance,
    VisibilityPolicy,
    _strip_reasoning,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.core.visibility import (
    REGISTRY,
    _consolidate_str_messages,
    _shuffle_simultaneous,
    _wrap_opponent_turn,
    all_prior,
    build_generation_messages,
    completed_rounds_only,
    get_visible_messages,
    should_see_thinking,
)
from tinker_cookbook.renderers import Message


def _make_spec(
    schedule: tuple[TurnSlot, ...],
    think_visibility: dict[Role, ThinkVisibility] | None = None,
) -> DebateSpec:
    return make_spec(
        task_prompt="Which is bigger, 2 or 3?",
        answer_by_role={Role.DEBATER_A: "2", Role.DEBATER_B: "3"},
        schedule=schedule,
        think_visibility=think_visibility,
    )


def _make_state(
    spec: DebateSpec,
    transcript: tuple[Utterance, ...] = (),
    slot_index: int = 0,
    rounds_completed: int = 0,
) -> DebateState:
    return DebateState(
        spec=spec,
        slot_index=slot_index,
        rounds_completed=rounds_completed,
        transcript=transcript,
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


# --- Registry ---


def test_registry_has_both_policies():
    assert VisibilityPolicy.ALL_PRIOR in REGISTRY
    assert VisibilityPolicy.COMPLETED_ROUNDS_ONLY in REGISTRY


# --- Reasoning stripping (dual-read <think>/<thinking>) ---


def test_strip_reasoning_removes_thinking_tags():
    assert _strip_reasoning("Hello <thinking>secret</thinking> world") == "Hello  world"


def test_strip_reasoning_removes_think_tags():
    assert _strip_reasoning("Hello <think>secret</think> world") == "Hello  world"


def test_strip_reasoning_multiline():
    assert _strip_reasoning("Before <thinking>\nline1\nline2\n</thinking> After") == "Before  After"
    assert _strip_reasoning("Before <think>\nline1\n</think> After") == "Before  After"


def test_strip_reasoning_no_tags():
    assert _strip_reasoning("No reasoning here") == "No reasoning here"


# --- all_prior policy (V2: returns list[Utterance]) ---


def test_all_prior_empty_transcript():
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1)
    spec = _make_spec(schedule)
    state = _make_state(spec)
    utts = all_prior(state, Role.DEBATER_A)
    assert utts == []


def test_all_prior_returns_all_utterances():
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, 2)
    spec = _make_spec(schedule)
    transcript = (
        _utt(Role.DEBATER_A, 0, "A says hello"),
        _utt(Role.DEBATER_B, 0, "B says hello"),
    )
    state = _make_state(spec, transcript=transcript, slot_index=2, rounds_completed=1)
    utts = all_prior(state, Role.DEBATER_A)
    assert len(utts) == 2
    assert utts[0].role == Role.DEBATER_A
    assert utts[0].text == "A says hello"
    assert utts[1].role == Role.DEBATER_B
    assert utts[1].text == "B says hello"


def test_all_prior_viewer_independent():
    """all_prior returns the same utterances regardless of viewer (viewer-dependent wrapping is done later)."""
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1)
    spec = _make_spec(schedule)
    transcript = (
        _utt(Role.DEBATER_A, 0, "A says hello"),
        _utt(Role.DEBATER_B, 0, "B says hello"),
    )
    state = _make_state(spec, transcript=transcript, slot_index=2, rounds_completed=1)
    utts_a = all_prior(state, Role.DEBATER_A)
    utts_b = all_prior(state, Role.DEBATER_B)
    assert utts_a == utts_b


# --- completed_rounds_only policy (V2: returns list[Utterance]) ---


def test_completed_rounds_only_excludes_current_round():
    schedule = build_schedule(ProtocolKind.SIMULTANEOUS, 2)
    spec = _make_spec(schedule)
    transcript = (
        _utt(Role.DEBATER_A, 0, "round 0 A"),
        _utt(Role.DEBATER_B, 0, "round 0 B"),
        _utt(Role.DEBATER_A, 1, "round 1 A"),
        _utt(Role.DEBATER_B, 1, "round 1 B"),
    )
    state = _make_state(spec, transcript=transcript, slot_index=1, rounds_completed=1)
    utts = completed_rounds_only(state, Role.DEBATER_A)
    assert len(utts) == 2
    assert utts[0].text == "round 0 A"
    assert utts[1].text == "round 0 B"


def test_completed_rounds_only_empty_first_round():
    schedule = build_schedule(ProtocolKind.SIMULTANEOUS, 2)
    spec = _make_spec(schedule)
    state = _make_state(spec, rounds_completed=0)
    utts = completed_rounds_only(state, Role.DEBATER_A)
    assert utts == []


# --- _wrap_opponent_turn ---


def test_wrap_opponent_turn_format():
    utt = _utt(Role.DEBATER_A, 0, "my argument")
    wrapped = _wrap_opponent_turn(utt, see_thinking=True)
    assert '<opponent_turn agent="A" phase="propose">' in wrapped
    assert "my argument" in wrapped
    assert "</opponent_turn>" in wrapped


def test_wrap_opponent_turn_strips_reasoning_closed():
    utt = _utt(Role.DEBATER_B, 0, "visible <think>secret</think> also visible")
    wrapped = _wrap_opponent_turn(utt, see_thinking=False)
    assert "<think>" not in wrapped
    assert "visible" in wrapped


def test_wrap_opponent_turn_keeps_reasoning_open():
    utt = _utt(Role.DEBATER_B, 0, "visible <think>kept</think> also visible")
    wrapped = _wrap_opponent_turn(utt, see_thinking=True)
    assert "<think>" in wrapped


# --- _shuffle_simultaneous ---


def test_shuffle_noop_for_debaters():
    schedule = build_schedule(ProtocolKind.SIMULTANEOUS, 1)
    spec = _make_spec(schedule)
    utts = [_utt(Role.DEBATER_A, 0, "A", slot_id=0), _utt(Role.DEBATER_B, 0, "B", slot_id=0)]
    state = _make_state(spec, transcript=tuple(utts), slot_index=1, rounds_completed=1)
    result = _shuffle_simultaneous(utts, Role.DEBATER_A, state)
    assert result == utts  # no change for debater


def test_shuffle_deterministic_for_judge():
    schedule = build_schedule(ProtocolKind.SIMULTANEOUS, 1)
    spec = _make_spec(schedule)
    utts = [_utt(Role.DEBATER_A, 0, "A", slot_id=0), _utt(Role.DEBATER_B, 0, "B", slot_id=0)]
    state = _make_state(spec, transcript=tuple(utts), slot_index=1, rounds_completed=1)
    result1 = _shuffle_simultaneous(utts, Role.JUDGE, state)
    result2 = _shuffle_simultaneous(utts, Role.JUDGE, state)
    assert result1 == result2  # same seed → same order


# --- _consolidate_str_messages ---


def test_consolidate_empty():
    assert _consolidate_str_messages([]) == []


def test_consolidate_different_roles():
    msgs = [Message(role="user", content="a"), Message(role="assistant", content="b")]
    assert len(_consolidate_str_messages(msgs)) == 2


def test_consolidate_same_role_merges():
    msgs = [Message(role="user", content="a"), Message(role="user", content="b")]
    result = _consolidate_str_messages(msgs)
    assert len(result) == 1
    assert result[0]["content"] == "a\n\nb"


def test_consolidate_skips_system():
    msgs = [Message(role="system", content="a"), Message(role="system", content="b")]
    result = _consolidate_str_messages(msgs)
    assert len(result) == 2  # system messages not merged


# --- get_visible_messages (V2: system + question + transcript) ---


def test_get_visible_messages_structure():
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1)
    spec = _make_spec(schedule)
    state = _make_state(spec)
    msgs = get_visible_messages(state, Role.DEBATER_A)
    # V2: system (persona only) + question (with answer)
    assert msgs[0]["role"] == "system"
    assert "debater_a" in msgs[0]["content"]
    assert "Which is bigger" not in msgs[0]["content"]  # task_prompt NOT in system
    assert msgs[1]["role"] == "user"
    assert "Which is bigger" in msgs[1]["content"]
    assert "Your assigned position: 2" in msgs[1]["content"]


def test_get_visible_messages_system_for_b():
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1)
    spec = _make_spec(schedule)
    state = _make_state(spec)
    msgs = get_visible_messages(state, Role.DEBATER_B)
    assert "debater_b" in msgs[0]["content"]
    # Answer in question message, not system
    assert msgs[1]["role"] == "user"
    assert "Your assigned position: 3" in msgs[1]["content"]


def test_get_visible_messages_uses_slot_policy():
    seq_schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1)
    sim_schedule = build_schedule(ProtocolKind.SIMULTANEOUS, 2)
    seq_spec = _make_spec(seq_schedule)
    sim_spec = _make_spec(sim_schedule)
    transcript = (
        _utt(Role.DEBATER_A, 0, "round 0 A"),
        _utt(Role.DEBATER_B, 0, "round 0 B"),
    )
    # Sequential fallback: system + question + 2 utterances = 4
    seq_state = _make_state(seq_spec, transcript=transcript, slot_index=2, rounds_completed=1)
    seq_msgs = get_visible_messages(seq_state, Role.DEBATER_A)
    assert len(seq_msgs) == 4

    # Simultaneous: system + question + round 0 utterances = 4
    sim_state = _make_state(sim_spec, transcript=transcript, slot_index=1, rounds_completed=1)
    sim_msgs = get_visible_messages(sim_state, Role.DEBATER_A)
    assert len(sim_msgs) == 4


def test_get_visible_messages_system_for_judge():
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1, include_judge_turns=True)
    spec = _make_spec(schedule)
    state = _make_state(spec)
    msgs = get_visible_messages(state, Role.JUDGE)
    assert msgs[0]["role"] == "system"
    assert "judge" in msgs[0]["content"]
    assert "Evaluate" in msgs[0]["content"]
    assert "Defend answer" not in msgs[0]["content"]


def test_get_visible_messages_schedule_exhausted_fallback():
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1)
    spec = _make_spec(schedule)
    transcript = (
        _utt(Role.DEBATER_A, 0, "A final"),
        _utt(Role.DEBATER_B, 0, "B final"),
    )
    state = _make_state(spec, transcript=transcript, slot_index=len(schedule), rounds_completed=1)
    msgs = get_visible_messages(state, Role.DEBATER_A)
    # V2: system + question + 2 utterances = 4
    assert len(msgs) == 4
    assert msgs[0]["role"] == "system"


def test_get_visible_messages_opponent_wrapped():
    """Opponent turns are wrapped in <opponent_turn> XML."""
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1)
    spec = _make_spec(schedule)
    transcript = (
        _utt(Role.DEBATER_A, 0, "A argues"),
        _utt(Role.DEBATER_B, 0, "B argues"),
    )
    state = _make_state(spec, transcript=transcript, slot_index=2, rounds_completed=1)
    msgs = get_visible_messages(state, Role.DEBATER_A)
    # Own turn = assistant, raw
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["content"] == "A argues"
    # Opponent turn = user, wrapped
    assert msgs[3]["role"] == "user"
    assert "<opponent_turn" in msgs[3]["content"]
    assert 'agent="B"' in msgs[3]["content"]


# --- No assigned stance ---


def _make_spec_no_stance(
    schedule: tuple[TurnSlot, ...],
    think_visibility: dict[Role, ThinkVisibility] | None = None,
) -> DebateSpec:
    return make_spec(
        task_prompt="Which is bigger, 2 or 3?",
        answer_by_role=None,
        schedule=schedule,
        think_visibility=think_visibility,
    )


def test_system_message_no_assigned_stance():
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1)
    spec = _make_spec_no_stance(schedule)
    state = _make_state(spec)
    msgs = get_visible_messages(state, Role.DEBATER_A)
    assert msgs[0]["role"] == "system"
    # V2: no "Argue your position" in system — system is just role identity
    assert "debater_a" in msgs[0]["content"]
    assert "Defend answer" not in msgs[0]["content"]


def test_all_prior_no_assigned_stance():
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1)
    spec = _make_spec_no_stance(schedule)
    transcript = (
        _utt(Role.DEBATER_A, 0, "A argues"),
        _utt(Role.DEBATER_B, 0, "B argues"),
    )
    state = _make_state(spec, transcript=transcript, slot_index=2, rounds_completed=1)
    utts = all_prior(state, Role.DEBATER_A)
    assert len(utts) == 2
    assert utts[0].role == Role.DEBATER_A
    assert utts[0].text == "A argues"
    assert utts[1].role == Role.DEBATER_B
    assert utts[1].text == "B argues"


# --- build_generation_messages ---


def test_build_generation_messages_structure():
    """Full prompt: system + question + transcript + instruction."""
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1)
    spec = DebateSpec(
        debate_id="test",
        problem=DebateProblemSpec(
            task_prompt="Is 2>3?",
            scoring_mode=ScoringMode.MCQ,
            answer_by_role={Role.DEBATER_A: "yes", Role.DEBATER_B: "no"},
        ),
        schedule=schedule,
    )
    transcript = (_utt(Role.DEBATER_A, 0, "A argues", slot_id=0),)
    state = _make_state(spec, transcript=transcript, slot_index=1, rounds_completed=0)
    msgs, prefill = build_generation_messages(state, Role.DEBATER_B)
    # Exact sequence: system(0), question(1), opponent transcript(2)
    # No instruction appended because default.yaml debater_b user template is empty
    # and no think/fields for debaters.
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"  # question
    assert "Is 2>3?" in msgs[1]["content"]
    assert "Your assigned position: no" in msgs[1]["content"]
    # Transcript: opponent turn wrapped in XML
    assert msgs[2]["role"] == "user"
    assert '<opponent_turn agent="A"' in msgs[2]["content"]
    assert "</opponent_turn>" in msgs[2]["content"]
    assert prefill is None


def test_build_generation_messages_judge_all_wrapped():
    """Judge view: ALL debate turns wrapped in <opponent_turn>."""
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1, include_judge_turns=True)
    spec = _make_spec(schedule)
    transcript = (
        _utt(Role.DEBATER_A, 0, "A argues"),
        _utt(Role.DEBATER_B, 0, "B argues"),
    )
    state = _make_state(spec, transcript=transcript, slot_index=2, rounds_completed=1)
    msgs, _prefill = build_generation_messages(state, Role.JUDGE, trigger="final")
    # Skip system(0) and question(1). Transcript may be consolidated into one user msg.
    transcript_content = "\n".join(str(m.get("content", "")) for m in msgs[2:])
    # Both debate turns must be wrapped (judge is never "self")
    assert transcript_content.count("<opponent_turn") == 2
    assert transcript_content.count("</opponent_turn>") == 2
    assert 'agent="A"' in transcript_content
    assert 'agent="B"' in transcript_content
    # No raw unwrapped turn content outside of opponent_turn tags
    assert "A argues" in transcript_content
    assert "B argues" in transcript_content


def test_build_generation_messages_question_not_consolidated():
    """Question message stays separate from first transcript message (scoped consolidation)."""
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1, include_judge_turns=True)
    spec = _make_spec(schedule)
    transcript = (
        _utt(Role.DEBATER_A, 0, "A argues"),
        _utt(Role.DEBATER_B, 0, "B argues"),
    )
    state = _make_state(spec, transcript=transcript, slot_index=2, rounds_completed=1)
    msgs, _ = build_generation_messages(state, Role.JUDGE, trigger="final")
    # Question is at index 1, should NOT be merged with transcript
    question_msg = msgs[1]
    assert question_msg["role"] == "user"
    assert "Which is bigger" in question_msg["content"]
    # Question must NOT contain opponent_turn XML (would indicate merge)
    assert "<opponent_turn" not in question_msg["content"]


def test_build_generation_messages_with_prefill():
    """build_generation_messages returns non-None prefill when configured."""
    import tempfile
    from tinker_cookbook.recipes.multiplayer_rl.debate.prompts import resolve_prompts as _rp

    yaml_content = """\
version: 2
system:
  judge:
    default: "judge"
  debater_a:
    default: "a"
  debater_b:
    default: "b"
question:
  debater_a: "q"
  debater_b: "q"
prefill:
  debater_a:
    default: "<think>"
"""
    f = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False)
    f.write(yaml_content)
    f.flush()
    f.close()
    _rp.cache_clear()
    try:
        spec = DebateSpec(
            debate_id="test",
            problem=DebateProblemSpec(
                task_prompt="Question?",
                scoring_mode=ScoringMode.MCQ,
                answer_by_role={Role.DEBATER_A: "yes", Role.DEBATER_B: "no"},
            ),
            schedule=build_schedule(ProtocolKind.SEQUENTIAL, 1),
            prompts_ref=f.name,
        )
        state = _make_state(spec)
        msgs, prefill = build_generation_messages(state, Role.DEBATER_A)
        assert prefill == "<think>"
        # Debater B has no prefill config -> None
        msgs_b, prefill_b = build_generation_messages(state, Role.DEBATER_B)
        assert prefill_b is None
    finally:
        import os

        os.unlink(f.name)
        _rp.cache_clear()


# --- opponent_wrap ---


def test_wrap_opponent_turn_fallback_no_prompts():
    """Without prompts/viewer, _wrap_opponent_turn uses hardcoded format."""
    utt = _utt(Role.DEBATER_A, 0, "hello")
    wrapped = _wrap_opponent_turn(utt, see_thinking=True)
    assert '<opponent_turn agent="A" phase="propose">' in wrapped
    assert "hello" in wrapped


def test_wrap_opponent_turn_with_opponent_wrap():
    """With prompts.opponent_wrap, _wrap_opponent_turn delegates to template."""
    import tempfile
    from tinker_cookbook.recipes.multiplayer_rl.debate.prompts import resolve_prompts as _rp

    yaml_content = """\
version: 2
system:
  judge:
    default: "judge"
  debater_a:
    default: "a"
  debater_b:
    default: "b"
question:
  debater_a: "q"
  debater_b: "q"
opponent_wrap:
  debater: "{{ text }}"
  judge: '<response from="Expert {{ label }}">\n{{ text }}\n</response>'
"""
    f = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False)
    f.write(yaml_content)
    f.flush()
    f.close()
    _rp.cache_clear()
    try:
        prompts = _rp(f.name)
        utt = _utt(Role.DEBATER_A, 0, "my argument")
        # Debater viewer: raw text
        wrapped_debater = _wrap_opponent_turn(
            utt, see_thinking=True, prompts=prompts, viewer=Role.DEBATER_B
        )
        assert wrapped_debater == "my argument"
        assert "<opponent_turn" not in wrapped_debater
        # Judge viewer: wrapped with custom template
        wrapped_judge = _wrap_opponent_turn(
            utt, see_thinking=True, prompts=prompts, viewer=Role.JUDGE
        )
        assert 'from="Expert A"' in wrapped_judge
        assert "my argument" in wrapped_judge
    finally:
        import os

        os.unlink(f.name)
        _rp.cache_clear()


def test_get_visible_messages_with_opponent_wrap():
    """Full integration: opponent_wrap templates used in get_visible_messages."""
    import tempfile
    from tinker_cookbook.recipes.multiplayer_rl.debate.prompts import resolve_prompts as _rp
    from tinker_cookbook.recipes.multiplayer_rl.debate.core.schedule import build_schedule as _bs

    yaml_content = """\
version: 2
system:
  judge:
    default: "judge"
  debater_a:
    default: "a"
  debater_b:
    default: "b"
question:
  debater_a: "q"
  debater_b: "q"
  judge: "q"
opponent_wrap:
  debater: "{{ text }}"
  judge: '<response from="Expert {{ label }}">\n{{ text }}\n</response>'
"""
    f = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False)
    f.write(yaml_content)
    f.flush()
    f.close()
    _rp.cache_clear()
    try:
        schedule = _bs(ProtocolKind.SEQUENTIAL, 1, include_judge_turns=True)
        spec = DebateSpec(
            debate_id="test",
            problem=DebateProblemSpec(
                task_prompt="Q?",
                scoring_mode=ScoringMode.MCQ,
                answer_by_role={Role.DEBATER_A: "yes", Role.DEBATER_B: "no"},
            ),
            schedule=schedule,
            prompts_ref=f.name,
        )
        transcript = (
            _utt(Role.DEBATER_A, 0, "A argues"),
            _utt(Role.DEBATER_B, 0, "B argues"),
        )
        state = _make_state(spec, transcript=transcript, slot_index=2, rounds_completed=1)
        # Debater B sees A's turn as raw text
        msgs_b = get_visible_messages(state, Role.DEBATER_B)
        opponent_msg = [
            m for m in msgs_b if m["role"] == "user" and "A argues" in str(m.get("content", ""))
        ]
        assert len(opponent_msg) == 1
        assert "<opponent_turn" not in opponent_msg[0]["content"]
        assert opponent_msg[0]["content"] == "A argues"
        # Judge sees wrapped
        msgs_j = get_visible_messages(state, Role.JUDGE)
        judge_content = "\n".join(str(m.get("content", "")) for m in msgs_j)
        assert 'from="Expert A"' in judge_content
        assert 'from="Expert B"' in judge_content
    finally:
        import os

        os.unlink(f.name)
        _rp.cache_clear()


# --- should_see_thinking ---


def test_should_see_thinking_disabled():
    """DISABLED: nobody sees thinking."""
    assert not should_see_thinking(ThinkVisibility.DISABLED, Role.DEBATER_A, Role.DEBATER_A)
    assert not should_see_thinking(ThinkVisibility.DISABLED, Role.DEBATER_A, Role.DEBATER_B)
    assert not should_see_thinking(ThinkVisibility.DISABLED, Role.DEBATER_A, Role.JUDGE)


def test_should_see_thinking_private():
    """PRIVATE: only the speaker sees their own thinking."""
    assert should_see_thinking(ThinkVisibility.PRIVATE, Role.DEBATER_A, Role.DEBATER_A)
    assert not should_see_thinking(ThinkVisibility.PRIVATE, Role.DEBATER_A, Role.DEBATER_B)
    assert not should_see_thinking(ThinkVisibility.PRIVATE, Role.DEBATER_A, Role.JUDGE)


def test_should_see_thinking_visible_to_judge():
    """VISIBLE_TO_JUDGE: speaker + judge see thinking."""
    assert should_see_thinking(ThinkVisibility.VISIBLE_TO_JUDGE, Role.DEBATER_A, Role.DEBATER_A)
    assert not should_see_thinking(ThinkVisibility.VISIBLE_TO_JUDGE, Role.DEBATER_A, Role.DEBATER_B)
    assert should_see_thinking(ThinkVisibility.VISIBLE_TO_JUDGE, Role.DEBATER_A, Role.JUDGE)


def test_should_see_thinking_open():
    """OPEN: everyone sees thinking."""
    assert should_see_thinking(ThinkVisibility.OPEN, Role.DEBATER_A, Role.DEBATER_A)
    assert should_see_thinking(ThinkVisibility.OPEN, Role.DEBATER_A, Role.DEBATER_B)
    assert should_see_thinking(ThinkVisibility.OPEN, Role.DEBATER_A, Role.JUDGE)
