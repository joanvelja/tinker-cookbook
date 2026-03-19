"""Stress test 7: Serialization round-trip for think_visibility.

Verifies _encode_spec / _decode_spec preserve think_visibility,
and that the encoded format uses string keys/values (not enum objects).
"""
import sys
sys.path.insert(0, ".")

from tinker_cookbook.recipes.multiplayer_rl.debate.eval.inspect_task import (
    _encode_spec,
    _decode_spec,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.core.schedule import build_schedule
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateProblemSpec,
    DebateSpec,
    ProtocolKind,
    Role,
    ScoringMode,
    ThinkVisibility,
)


def test_serialization_roundtrip():
    print("Testing serialization round-trip...")

    original_tv = {
        Role.DEBATER_A: ThinkVisibility.VISIBLE_TO_JUDGE,
        Role.DEBATER_B: ThinkVisibility.PRIVATE,
        Role.JUDGE: ThinkVisibility.DISABLED,
    }

    spec = DebateSpec(
        debate_id="roundtrip_test",
        problem=DebateProblemSpec(
            task_prompt="Test question",
            scoring_mode=ScoringMode.MCQ,
            answer_by_role={Role.DEBATER_A: "A", Role.DEBATER_B: "B"},
        ),
        schedule=build_schedule(ProtocolKind.SEQUENTIAL, 2),
        think_visibility=original_tv,
        protocol_kind=ProtocolKind.SEQUENTIAL,
        prompts_ref="default",
    )

    # Encode
    encoded = _encode_spec(spec)

    # Verify encoded format uses string keys/values (not enum objects)
    encoded_tv = encoded["think_visibility"]
    for k, v in encoded_tv.items():
        assert isinstance(k, str), f"Encoded key should be str, got {type(k)}: {k}"
        assert isinstance(v, str), f"Encoded value should be str, got {type(v)}: {v}"
    print(f"  Encoded think_visibility: {encoded_tv}")

    # Verify expected encoded values
    assert encoded_tv["debater_a"] == "visible_to_judge", f"Expected 'visible_to_judge', got {encoded_tv['debater_a']}"
    assert encoded_tv["debater_b"] == "private", f"Expected 'private', got {encoded_tv['debater_b']}"
    assert encoded_tv["judge"] == "disabled", f"Expected 'disabled', got {encoded_tv['judge']}"

    # Decode
    decoded = _decode_spec(encoded)

    # Verify decoded matches original
    for role in [Role.DEBATER_A, Role.DEBATER_B, Role.JUDGE]:
        original_val = original_tv[role]
        decoded_val = decoded.think_visibility.get(role)
        assert decoded_val == original_val, (
            f"Mismatch for {role}: original={original_val}, decoded={decoded_val}"
        )

    print("  PASS: round-trip preserves think_visibility")


def test_empty_think_visibility():
    """Edge case: empty think_visibility round-trips correctly."""
    print("Testing empty think_visibility round-trip...")

    spec = DebateSpec(
        debate_id="empty_tv_test",
        problem=DebateProblemSpec(
            task_prompt="Test",
            scoring_mode=ScoringMode.MCQ,
        ),
        schedule=build_schedule(ProtocolKind.SEQUENTIAL, 1),
        think_visibility={},
        protocol_kind=ProtocolKind.SEQUENTIAL,
        prompts_ref="default",
    )

    encoded = _encode_spec(spec)
    assert encoded["think_visibility"] == {}, "Empty TV should encode to {}"

    decoded = _decode_spec(encoded)
    assert len(decoded.think_visibility) == 0, "Empty TV should decode to empty"

    print("  PASS: empty think_visibility round-trip")


if __name__ == "__main__":
    test_serialization_roundtrip()
    test_empty_think_visibility()
    print("\nStress test 7: ALL PASS")
    sys.exit(0)
