"""Stress test 8: conftest.make_spec verification.

Verifies:
1. make_spec() defaults -> all roles ThinkVisibility.DISABLED
2. make_spec(think_visibility=...) -> correct custom values
3. make_spec no longer accepts open_reasoning parameter
"""
import sys
import inspect
sys.path.insert(0, ".")

from tinker_cookbook.recipes.multiplayer_rl.debate.tests.conftest import make_spec
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    Role,
    ThinkVisibility,
)


def test_defaults():
    """make_spec() with defaults -> all roles have ThinkVisibility.DISABLED."""
    print("Testing make_spec defaults...")
    spec = make_spec()

    for role in [Role.DEBATER_A, Role.DEBATER_B, Role.JUDGE]:
        val = spec.think_visibility.get(role)
        assert val == ThinkVisibility.DISABLED, (
            f"Default think_visibility for {role} should be DISABLED, got {val}"
        )

    print("  PASS: defaults all DISABLED")


def test_custom_think_visibility():
    """make_spec(think_visibility=...) -> correct values."""
    print("Testing make_spec with custom think_visibility...")
    custom_tv = {
        Role.DEBATER_A: ThinkVisibility.PRIVATE,
        Role.DEBATER_B: ThinkVisibility.VISIBLE_TO_JUDGE,
        Role.JUDGE: ThinkVisibility.OPEN,
    }
    spec = make_spec(think_visibility=custom_tv)

    for role, expected in custom_tv.items():
        actual = spec.think_visibility.get(role)
        assert actual == expected, (
            f"think_visibility for {role}: expected {expected}, got {actual}"
        )

    print("  PASS: custom think_visibility correct")


def test_no_open_reasoning_param():
    """make_spec no longer accepts open_reasoning parameter."""
    print("Testing make_spec rejects open_reasoning...")

    sig = inspect.signature(make_spec)
    assert "open_reasoning" not in sig.parameters, (
        f"make_spec should NOT have open_reasoning param. "
        f"Found params: {list(sig.parameters.keys())}"
    )

    # Also verify it actually raises on the kwarg
    try:
        make_spec(open_reasoning=True)
        assert False, "make_spec(open_reasoning=True) should raise TypeError"
    except TypeError:
        pass  # Expected

    print("  PASS: open_reasoning rejected")


if __name__ == "__main__":
    test_defaults()
    test_custom_think_visibility()
    test_no_open_reasoning_param()
    print("\nStress test 8: ALL PASS")
    sys.exit(0)
