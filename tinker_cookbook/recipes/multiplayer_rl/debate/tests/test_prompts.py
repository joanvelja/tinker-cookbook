"""Tests for the YAML-driven prompt system (V2)."""

from __future__ import annotations

import logging
import tempfile

import pytest

from tinker_cookbook.recipes.multiplayer_rl.debate.prompts import (
    check_ab_symmetry,
    resolve_prompts,
    _check_migration_lint,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.scoring.fields import FieldSpec
from tinker_cookbook.recipes.multiplayer_rl.debate.scoring.parsing import (
    generate_format_instructions,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.tests.conftest import make_spec
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateProblemSpec,
    DebateSpec,
    DebateState,
    Phase,
    ProtocolKind,
    Role,
    ScoringMode,
    TurnSlot,
    VisibilityPolicy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slot(slot_id: int, round_index: int, phase: Phase) -> TurnSlot:
    return TurnSlot(
        slot_id=slot_id,
        round_index=round_index,
        phase=phase,
        actors=(Role.DEBATER_A,),
        boundary_after=False,
        visibility_policy=VisibilityPolicy.ALL_PRIOR,
    )


def _make_schedule(num_rounds: int = 2) -> tuple[TurnSlot, ...]:
    slots: list[TurnSlot] = []
    sid = 0
    for r in range(num_rounds):
        phase = Phase.PROPOSE if r == 0 else Phase.CRITIQUE
        slots.append(_slot(sid, r, phase))
        sid += 1
        slots.append(
            TurnSlot(
                slot_id=sid,
                round_index=r,
                phase=phase,
                actors=(Role.DEBATER_B,),
                boundary_after=True,
                visibility_policy=VisibilityPolicy.ALL_PRIOR,
            )
        )
        sid += 1
    return tuple(slots)


def _make_spec(
    *,
    task_prompt: str = "Is P=NP?",
    answer_by_role: dict[Role, str] | None = None,
    num_rounds: int = 2,
    open_reasoning: bool = False,
    prompts_ref: str = "default",
) -> DebateSpec:
    if answer_by_role is None:
        answer_by_role = {Role.DEBATER_A: "Yes", Role.DEBATER_B: "No"}
    return make_spec(
        task_prompt=task_prompt,
        answer_by_role=answer_by_role,
        schedule=_make_schedule(num_rounds),
        open_reasoning=open_reasoning,
        prompts_ref=prompts_ref,
    )


def _make_state(
    spec: DebateSpec | None = None,
    slot_index: int = 0,
) -> DebateState:
    if spec is None:
        spec = _make_spec()
    return DebateState(
        spec=spec,
        slot_index=slot_index,
        rounds_completed=0,
        transcript=(),
        pending_simultaneous={},
        judge_trace=(),
        done=False,
        outcome=None,
    )


_tmp_files: list[str] = []


def _tmp_yaml(content: str) -> str:
    """Write content to a temp YAML file and return its path."""
    f = tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False)
    f.write(content)
    f.flush()
    f.close()
    _tmp_files.append(f.name)
    return f.name


# Minimal valid V2 YAML for inline tests that need custom templates.
_V2_MINIMAL = """\
version: 2
system:
  judge:
    default: "{judge_tmpl}"
  debater_a:
    default: "{da_tmpl}"
  debater_b:
    default: "{db_tmpl}"
question:
  debater_a: "q"
  debater_b: "q"
"""


def _v2_yaml(
    judge_sys: str = "judge",
    da_sys: str = "a",
    db_sys: str = "b",
    extra: str = "",
) -> str:
    return f"""\
version: 2
system:
  judge:
    default: "{judge_sys}"
  debater_a:
    default: "{da_sys}"
  debater_b:
    default: "{db_sys}"
question:
  debater_a: "q"
  debater_b: "q"
{extra}"""


# ---------------------------------------------------------------------------
# Fixture: clear LRU cache between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cache():
    resolve_prompts.cache_clear()
    yield
    resolve_prompts.cache_clear()
    # Clean up temp files
    import os

    while _tmp_files:
        try:
            os.unlink(_tmp_files.pop())
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Loading & caching
# ---------------------------------------------------------------------------


def test_resolve_default():
    """Default prompts: system is persona-only, question has task+answer."""
    p = resolve_prompts("default")
    state = _make_state()

    # Judge system — no task_prompt
    judge_sys = p.render_system(state, Role.JUDGE)
    assert judge_sys == "You are the judge. Evaluate the debaters' arguments."

    # Debater A system — persona only
    da_sys = p.render_system(state, Role.DEBATER_A)
    assert da_sys == "You are debater_a. Your goal is to defend your assigned position."

    # Debater B system — persona only
    db_sys = p.render_system(state, Role.DEBATER_B)
    assert db_sys == "You are debater_b. Your goal is to defend your assigned position."

    # Question has task_prompt + assigned answer
    da_q = p.render_question(state, Role.DEBATER_A)
    assert da_q == "Is P=NP?\n\nYour assigned position: Yes"
    db_q = p.render_question(state, Role.DEBATER_B)
    assert db_q == "Is P=NP?\n\nYour assigned position: No"

    judge_q = p.render_question(state, Role.JUDGE)
    assert judge_q == "Is P=NP?"


def test_resolve_caching():
    """Same ref returns the same object (LRU cache identity)."""
    p1 = resolve_prompts("default")
    p2 = resolve_prompts("default")
    assert p1 is p2


def test_resolve_file_not_found():
    """Non-existent ref raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        resolve_prompts("nonexistent_prompt_ref_xyz")


def test_content_hash_deterministic():
    """Same YAML content produces the same hash across loads."""
    p1 = resolve_prompts("default")
    resolve_prompts.cache_clear()
    p2 = resolve_prompts("default")
    assert p1.content_hash == p2.content_hash
    assert len(p1.content_hash) == 64  # SHA256 hex


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_v1_rejected():
    """Version 1 YAML is rejected with a clear error."""
    path = _tmp_yaml("version: 1\nsystem:\n  judge:\n    default: hi\nuser: {}\n")
    with pytest.raises(ValueError, match="expected 2"):
        resolve_prompts(path)


def test_missing_question_section():
    """Missing debater question entry raises ValueError."""
    path = _tmp_yaml("""\
version: 2
system:
  judge:
    default: "j"
  debater_a:
    default: "a"
  debater_b:
    default: "b"
question:
  debater_a: "q"
""")
    with pytest.raises(ValueError, match="missing required role 'debater_b'"):
        resolve_prompts(path)


def test_unknown_keys_rejected():
    """Typo like 'debaterA' in role keys fails validation."""
    path = _tmp_yaml(_v2_yaml(extra="").replace("debater_a", "debaterA"))
    with pytest.raises(ValueError):
        resolve_prompts(path)


def test_invalid_field_tag_name():
    """Field name with spaces or special chars is rejected."""
    path = _tmp_yaml(
        _v2_yaml(
            extra="""\
fields:
  judge:
    final:
      "bad name!":
        type: str
        description: "oops"
"""
        )
    )
    with pytest.raises(ValueError, match="Invalid field tag name"):
        resolve_prompts(path)


def test_strict_undefined():
    """Template referencing unknown variable fails at render time (StrictUndefined)."""
    path = _tmp_yaml(_v2_yaml(judge_sys="{{nonexistent_var}}"))
    p = resolve_prompts(path)
    state = _make_state()
    with pytest.raises(Exception):  # jinja2.UndefinedError
        p.render_system(state, Role.JUDGE)


def test_migration_lint_catches_reasoning_instruction():
    """Migration lint detects reasoning_instruction in templates before compilation."""
    path = _tmp_yaml("""\
version: 2
system:
  judge:
    default: "{{ reasoning_instruction }}"
  debater_a:
    default: "ok"
  debater_b:
    default: "ok"
question:
  debater_a: "q"
  debater_b: "q"
""")
    with pytest.raises(ValueError, match="removed variable 'reasoning_instruction'"):
        resolve_prompts(path)


def test_migration_lint_direct():
    """_check_migration_lint catches reasoning_instruction in nested blocks."""
    with pytest.raises(ValueError, match="reasoning_instruction"):
        _check_migration_lint({"user": {"debater_a": {"default": "{{ reasoning_instruction }}"}}})
    # Clean block should not raise
    _check_migration_lint({"system": {"judge": {"default": "hello"}}})


# ---------------------------------------------------------------------------
# Phase lookup & fallback
# ---------------------------------------------------------------------------


def test_phase_expanded_from_default():
    """'default' is expanded to all phases at compile time — no runtime fallback."""
    p = resolve_prompts("default")
    spec = _make_spec(num_rounds=2)
    # slot_index=2 is round 1, phase=CRITIQUE — expanded from 'default'
    state = _make_state(spec=spec, slot_index=2)
    sys_msg = p.render_system(state, Role.DEBATER_A)
    assert "debater_a" in sys_msg
    q_msg = p.render_question(state, Role.DEBATER_A)
    assert "Your assigned position: Yes" in q_msg


def test_done_state_renders():
    """Schedule-exhausted (slot_index past end) renders via expanded 'done' key."""
    p = resolve_prompts("default")
    spec = _make_spec(num_rounds=1)
    # 1 round = 2 slots (A, B), so slot_index=2 is past the end → phase="done"
    state = _make_state(spec=spec, slot_index=2)
    sys_msg = p.render_system(state, Role.JUDGE)
    assert "judge" in sys_msg.lower()


# ---------------------------------------------------------------------------
# Context keys
# ---------------------------------------------------------------------------


def test_judge_context():
    """Judge gets answer=='', answer_a/answer_b populated."""
    p = resolve_prompts("default")
    state = _make_state()
    # Judge system should not contain "Yes" or "No" (those are debater answers)
    sys_msg = p.render_system(state, Role.JUDGE)
    assert "Yes" not in sys_msg
    assert "No" not in sys_msg
    assert "judge" in sys_msg.lower()


def test_answer_by_role_none():
    """Graceful handling when answer_by_role is None."""
    p = resolve_prompts("default")
    spec = DebateSpec(
        debate_id="test",
        problem=DebateProblemSpec(
            task_prompt="Question?",
            scoring_mode=ScoringMode.MCQ,
            answer_by_role=None,
        ),
        schedule=_make_schedule(1),
        open_reasoning=False,
        protocol_kind=ProtocolKind.SEQUENTIAL,
    )
    state = _make_state(spec=spec)
    # V2: system has no answer info, just persona
    da_sys = p.render_system(state, Role.DEBATER_A)
    assert "You are debater_a." in da_sys
    assert "assigned position" not in da_sys
    # Question: no assigned answer block
    da_q = p.render_question(state, Role.DEBATER_A)
    assert da_q == "Question?"


def test_is_first_last_round():
    """is_first_round and is_last_round correct at boundaries."""
    path = _tmp_yaml(
        _v2_yaml(
            da_sys="first={{is_first_round}} last={{is_last_round}}",
            db_sys="first={{is_first_round}} last={{is_last_round}}",
        )
    )
    p = resolve_prompts(path)
    spec = _make_spec(num_rounds=3)

    # Round 0, slot 0 -> first=True, last=False
    state0 = _make_state(spec=spec, slot_index=0)
    msg0 = p.render_system(state0, Role.DEBATER_A)
    assert msg0 == "first=True last=False"

    # Round 2 (last), slot 4 -> first=False, last=True
    state_last = _make_state(spec=spec, slot_index=4)
    msg_last = p.render_system(state_last, Role.DEBATER_A)
    assert msg_last == "first=False last=True"


# ---------------------------------------------------------------------------
# User templates
# ---------------------------------------------------------------------------


def test_user_template_empty_returns_none():
    """Empty user template returns None (skip message)."""
    p = resolve_prompts("default")
    state = _make_state()
    result = p.render_user(state, Role.DEBATER_A)
    assert result is None


def test_user_template_with_content():
    """Non-empty user template returns rendered string."""
    p = resolve_prompts("default")
    state = _make_state()
    result = p.render_user(state, Role.JUDGE, trigger="final")
    assert result is not None
    assert "convincing" in result


# ---------------------------------------------------------------------------
# Question rendering
# ---------------------------------------------------------------------------


def test_render_question_per_role():
    """render_question returns correct content per role."""
    p = resolve_prompts("default")
    state = _make_state()

    judge_q = p.render_question(state, Role.JUDGE)
    assert judge_q == "Is P=NP?"

    da_q = p.render_question(state, Role.DEBATER_A)
    assert "Is P=NP?" in da_q
    assert "Your assigned position: Yes" in da_q

    db_q = p.render_question(state, Role.DEBATER_B)
    assert "Your assigned position: No" in db_q


def test_render_question_absent_role():
    """render_question returns None for roles not in question section."""
    path = _tmp_yaml(_v2_yaml())  # no judge in question by default from _v2_yaml
    p = resolve_prompts(path)
    state = _make_state()
    # judge not in question section of _v2_yaml
    assert p.render_question(state, Role.JUDGE) is None


# ---------------------------------------------------------------------------
# Think instruction
# ---------------------------------------------------------------------------


def test_think_true_closed():
    """think: true with closed reasoning -> private instruction."""
    path = _tmp_yaml(_v2_yaml(extra="think:\n  debater_a: true\n"))
    p = resolve_prompts(path)
    state = _make_state(_make_spec(open_reasoning=False, prompts_ref=path))
    instruction = p.get_think_instruction(state, Role.DEBATER_A)
    assert instruction is not None
    assert "private" in instruction.lower() or "will NOT see" in instruction


def test_think_true_open():
    """think: true with open reasoning -> visible instruction."""
    path = _tmp_yaml(_v2_yaml(extra="think:\n  debater_a: true\n"))
    p = resolve_prompts(path)
    spec = _make_spec(open_reasoning=True, prompts_ref=path)
    state = _make_state(spec)
    instruction = p.get_think_instruction(state, Role.DEBATER_A)
    assert instruction is not None
    assert "visible" in instruction.lower()


def test_think_false():
    """think: false -> None."""
    path = _tmp_yaml(_v2_yaml(extra="think:\n  debater_a: false\n"))
    p = resolve_prompts(path)
    state = _make_state()
    assert p.get_think_instruction(state, Role.DEBATER_A) is None


def test_think_custom_string():
    """think with custom string -> rendered template."""
    path = _tmp_yaml(_v2_yaml(extra='think:\n  debater_a: "Think step by step about {{phase}}."\n'))
    p = resolve_prompts(path)
    state = _make_state()
    instruction = p.get_think_instruction(state, Role.DEBATER_A)
    assert instruction is not None
    assert "Think step by step about propose" in instruction


def test_think_absent():
    """No think section -> None for all roles."""
    p = resolve_prompts("default")
    state = _make_state()
    assert p.get_think_instruction(state, Role.DEBATER_A) is None
    assert p.get_think_instruction(state, Role.JUDGE) is None


def test_think_per_phase():
    """Per-phase think: {propose: true, critique: false} -> critique returns None."""
    path = _tmp_yaml(
        _v2_yaml(
            extra="""\
think:
  debater_a:
    propose: true
    critique: false
"""
        )
    )
    p = resolve_prompts(path)
    spec = _make_spec(num_rounds=2)

    # slot 0 = propose -> explicit true
    state_propose = _make_state(spec=spec, slot_index=0)
    assert p.get_think_instruction(state_propose, Role.DEBATER_A) is not None

    # slot 2 = critique -> explicit false
    state_critique = _make_state(spec=spec, slot_index=2)
    assert p.get_think_instruction(state_critique, Role.DEBATER_A) is None


def test_render_user_empty_phase_but_think_true():
    """Empty phase template + think=true -> returns think instruction, NOT None."""
    path = _tmp_yaml(
        _v2_yaml(
            extra="""\
user:
  debater_a:
    default: ""
think:
  debater_a: true
"""
        )
    )
    p = resolve_prompts(path)
    state = _make_state()
    result = p.render_user(state, Role.DEBATER_A)
    assert result is not None
    assert "<thinking>" in result


def test_render_user_all_three_parts():
    """render_user assembles phase + think + fields when all are present."""
    path = _tmp_yaml(
        _v2_yaml(
            extra="""\
user:
  judge:
    final: "Render your verdict."
  debater_a:
    default: ""
  debater_b:
    default: ""
fields:
  judge:
    final:
      decision:
        type: str
        description: "debater_a or debater_b or tie"
think:
  judge: true
"""
        )
    )
    p = resolve_prompts(path)
    state = _make_state()
    result = p.render_user(state, Role.JUDGE, trigger="final")
    assert result is not None
    # Phase instruction present
    assert "Render your verdict." in result
    # Think instruction present
    assert "<thinking>" in result
    # Field instruction present
    assert "<decision>" in result
    # Parts are separated by double-newlines
    parts = result.split("\n\n")
    assert len(parts) >= 3


# ---------------------------------------------------------------------------
# Prefill rendering
# ---------------------------------------------------------------------------


def test_render_prefill_absent():
    """No prefill section -> None."""
    p = resolve_prompts("default")
    state = _make_state()
    assert p.render_prefill(state, Role.DEBATER_A) is None


def test_render_prefill_with_config():
    """Prefill section renders correctly."""
    path = _tmp_yaml(
        _v2_yaml(
            extra="""\
prefill:
  debater_a:
    default: "<think>"
"""
        )
    )
    p = resolve_prompts(path)
    state = _make_state()
    result = p.render_prefill(state, Role.DEBATER_A)
    assert result == "<think>"


def test_render_prefill_empty_returns_none():
    """Prefill with empty string -> None."""
    path = _tmp_yaml(
        _v2_yaml(
            extra="""\
prefill:
  debater_a:
    default: "   "
"""
        )
    )
    p = resolve_prompts(path)
    state = _make_state()
    assert p.render_prefill(state, Role.DEBATER_A) is None


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------


def test_field_instructions_generated():
    """Judge fields produce correct XML instructions."""
    p = resolve_prompts("default")
    fi = p.get_field_instructions("judge", "final")
    assert fi is not None
    assert "<decision>" in fi
    assert "<reason>" in fi
    assert "You MUST include" in fi


def test_field_names_lookup():
    """get_field_names returns correct field names."""
    p = resolve_prompts("default")
    names = p.get_field_names("judge", "final")
    assert names == ["decision", "reason"]


def test_field_names_none_for_missing():
    """get_field_names returns None when no fields defined."""
    p = resolve_prompts("default")
    assert p.get_field_names("debater_a", "final") is None
    assert p.get_field_names("judge", "nonexistent") is None


def test_debater_fields_generate_instructions():
    """Debater phase fields produce format instructions when defined."""
    path = _tmp_yaml(
        _v2_yaml(
            extra="""\
user:
  debater_a:
    default: "your turn"
  debater_b:
    default: "your turn"
fields:
  debater_a:
    default:
      argument:
        type: str
        description: "your main argument"
  debater_b:
    default:
      argument:
        type: str
        description: "your main argument"
"""
        )
    )
    p = resolve_prompts(path)
    fi = p.get_field_instructions("debater_a", "default")
    assert fi is not None
    assert "<argument>" in fi


# ---------------------------------------------------------------------------
# Template injection prevention
# ---------------------------------------------------------------------------


def test_task_prompt_injection():
    """task_prompt containing {{ }} doesn't execute as Jinja."""
    p = resolve_prompts("default")
    malicious_prompt = "{{ 7 * 7 }} and {{answer}}"
    spec = _make_spec(task_prompt=malicious_prompt)
    state = _make_state(spec=spec)
    # In V2, task_prompt is in question, not system
    q_msg = p.render_question(state, Role.JUDGE)
    assert "{{ 7 * 7 }}" in q_msg
    assert "49" not in q_msg
    assert "{{answer}}" in q_msg


# ---------------------------------------------------------------------------
# A/B symmetry
# ---------------------------------------------------------------------------


def test_ab_symmetry_warning(caplog):
    """Warning emitted when debater_a and debater_b have asymmetric phase keys."""
    path = _tmp_yaml(
        _v2_yaml(
            extra="""\
user:
  debater_a:
    propose: "a"
    critique: "a critique"
  debater_b:
    propose: "b"
"""
        )
    )
    with caplog.at_level(logging.WARNING):
        p = resolve_prompts(path)

    assert any("A/B asymmetry" in r.message for r in caplog.records)
    warnings = check_ab_symmetry(p)
    assert len(warnings) == 1
    assert "debater_a" in warnings[0] and "debater_b" in warnings[0]


def test_ab_symmetry_clean():
    """No warnings when debater_a and debater_b are symmetric."""
    p = resolve_prompts("default")
    warnings = check_ab_symmetry(p)
    assert warnings == []


# ---------------------------------------------------------------------------
# generate_format_instructions standalone
# ---------------------------------------------------------------------------


def test_generate_field_instructions_format():
    """generate_format_instructions produces correct output."""
    fields = {
        "score": FieldSpec(type=float, description="0-10 rating"),
        "comment": FieldSpec(type=str, description="brief comment"),
    }
    result = generate_format_instructions(fields)
    assert result == (
        "You MUST include the following XML tags in your response:\n"
        "<score>0-10 rating</score>\n"
        "<comment>brief comment</comment>"
    )


# ---------------------------------------------------------------------------
# Think config validation
# ---------------------------------------------------------------------------


def test_think_invalid_type_rejected():
    """think with invalid type (list) is rejected."""
    path = _tmp_yaml(_v2_yaml(extra="think:\n  debater_a: [1, 2]\n"))
    with pytest.raises(ValueError, match="expected bool, str, or dict"):
        resolve_prompts(path)


# ---------------------------------------------------------------------------
# Default mixing rejection
# ---------------------------------------------------------------------------


def test_mixed_default_rejected_system():
    """'default' alongside phase-specific keys in system raises ValueError."""
    path = _tmp_yaml("""\
version: 2
system:
  judge:
    default: "j"
    propose: "j propose"
  debater_a:
    default: "a"
  debater_b:
    default: "b"
question:
  debater_a: "q"
  debater_b: "q"
""")
    with pytest.raises(ValueError, match="cannot coexist"):
        resolve_prompts(path)


def test_mixed_default_rejected_user():
    """'default' alongside phase-specific keys in user raises ValueError."""
    path = _tmp_yaml(
        _v2_yaml(
            extra="""\
user:
  debater_a:
    default: ""
    propose: "hello"
  debater_b:
    default: ""
"""
        )
    )
    with pytest.raises(ValueError, match="cannot coexist"):
        resolve_prompts(path)


def test_mixed_default_rejected_think():
    """'default' alongside phase-specific keys in think raises ValueError."""
    path = _tmp_yaml(
        _v2_yaml(extra="think:\n  debater_a:\n    default: true\n    critique: false\n")
    )
    with pytest.raises(ValueError, match="cannot coexist"):
        resolve_prompts(path)


# ---------------------------------------------------------------------------
# opponent_wrap
# ---------------------------------------------------------------------------


def test_opponent_wrap_parsed():
    """opponent_wrap YAML section loads correctly."""
    path = _tmp_yaml(
        _v2_yaml(
            extra="""\
opponent_wrap:
  debater: "{{ text }}"
  judge: '<response from="Expert {{ label }}">\n{{ text }}\n</response>'
"""
        )
    )
    p = resolve_prompts(path)
    assert p.opponent_wrap is not None
    assert "debater" in p.opponent_wrap
    assert "judge" in p.opponent_wrap


def test_opponent_wrap_absent():
    """No opponent_wrap section -> None (legacy fallback)."""
    p = resolve_prompts("default")
    assert p.opponent_wrap is None


def test_render_opponent_wrap_debater():
    """render_opponent_wrap with debater viewer -> debater template."""
    path = _tmp_yaml(
        _v2_yaml(
            extra="""\
opponent_wrap:
  debater: "{{ text }}"
  judge: '<response from="Expert {{ label }}">\n{{ text }}\n</response>'
"""
        )
    )
    p = resolve_prompts(path)
    result = p.render_opponent_wrap("hello world", "A", "propose", Role.DEBATER_B)
    assert result == "hello world"


def test_render_opponent_wrap_judge():
    """render_opponent_wrap with judge viewer -> judge template with label."""
    path = _tmp_yaml(
        _v2_yaml(
            extra="""\
opponent_wrap:
  debater: "{{ text }}"
  judge: '<response from="Expert {{ label }}">\n{{ text }}\n</response>'
"""
        )
    )
    p = resolve_prompts(path)
    result = p.render_opponent_wrap("my argument", "A", "propose", Role.JUDGE)
    assert 'from="Expert A"' in result
    assert "my argument" in result


def test_opponent_wrap_validation_bad_keys():
    """opponent_wrap with unknown keys is rejected."""
    path = _tmp_yaml(
        _v2_yaml(extra="opponent_wrap:\n  badkey: '{{ text }}'\n")
    )
    with pytest.raises(ValueError, match="unknown keys"):
        resolve_prompts(path)


def test_opponent_wrap_validation_non_string():
    """opponent_wrap with non-string value is rejected."""
    path = _tmp_yaml(
        _v2_yaml(extra="opponent_wrap:\n  debater: 42\n")
    )
    with pytest.raises(ValueError, match="expected str"):
        resolve_prompts(path)
