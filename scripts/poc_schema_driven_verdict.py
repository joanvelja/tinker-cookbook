"""PoC: Schema-driven verdict extraction — MDL solution for Bugs 1, 2, 3.

Demonstrates that adding EnumScoring to the YAML `decision` field lets us:
  - Discover field names from specs (kills Bug 2: reasoning vs reason)
  - Normalize verdicts via classify_enum (kills Bug 1: A/B vs debater_a/debater_b)
  - Use a single strip_think (kills Bug 3: truncated thinking)

Run: uv run python scripts/poc_schema_driven_verdict.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

# --- Add project root to path so we can import the actual field infra ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinker_cookbook.recipes.multiplayer_rl.debate.scoring.fields import (
    EnumScoring,
    FieldSpec,
    _resolve_fields,
    classify_enum,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.scoring.parsing import extract_fields
from tinker_cookbook.recipes.multiplayer_rl.debate.scoring.mcq import strip_think
from tinker_cookbook.recipes.multiplayer_rl.debate.types import Role

# ============================================================================
# 1. The YAML change: add EnumScoring to `decision`
# ============================================================================

# Current YAML (broken):
CURRENT_YAML_FIELDS = {
    "reasoning": {"type": "str"},
    "decision": {"type": "str", "description": "A, B, or tie"},
}

# MDL YAML (fixed): decision gets enum scoring
MDL_YAML_FIELDS = {
    "reasoning": {"type": "str"},
    "decision": {
        "type": "str",
        "description": "A, B, or tie",
        "scoring": {"mode": "enum", "values": ["A", "B", "tie"]},
    },
}

# ============================================================================
# 2. Schema-driven field discovery (replaces hardcoded "reason"/"decision")
# ============================================================================

# The ONLY hardcoded mapping: canonical enum values → Roles.
# These are the values WE define in the YAML, not raw LLM output.
_ENUM_TO_ROLE: dict[str, Role | None] = {
    "A": Role.DEBATER_A,
    "B": Role.DEBATER_B,
    "tie": None,
}


def discover_verdict_fields(
    specs: dict[str, FieldSpec],
) -> tuple[str | None, str | None]:
    """Find the decision field (has EnumScoring) and reasoning field (str, no scoring).

    Returns (decision_key, reasoning_key). Either may be None if not found.
    """
    decision_key = None
    reasoning_key = None
    for name, spec in specs.items():
        if isinstance(spec.scoring, EnumScoring):
            decision_key = name
        elif spec.type is str and spec.scoring is None:
            reasoning_key = name
    return decision_key, reasoning_key


@dataclass(frozen=True)
class VerdictResult:
    winner: Role | None
    reason: str
    raw_decision: str
    normalized_decision: str | None


def parse_verdict_from_specs(
    text: str,
    specs: dict[str, FieldSpec],
) -> VerdictResult:
    """Schema-driven verdict parsing. No hardcoded field names or value mappings."""
    # Step 1: strip thinking (single canonical function, handles unclosed tags)
    cleaned, _reasoning = strip_think(text)

    # Step 2: extract fields using schema derived from specs
    fields = extract_fields(cleaned, specs)
    if fields is None:
        fields = {}

    # Step 3: discover which field is the decision, which is the reasoning
    decision_key, reasoning_key = discover_verdict_fields(specs)

    # Step 4: get raw values using discovered keys
    raw_decision = str(fields.get(decision_key, "")) if decision_key else ""
    reason = str(fields.get(reasoning_key, "")) if reasoning_key else ""

    # Step 5: normalize decision via the spec's EnumScoring classifier
    normalized = None
    if decision_key and raw_decision:
        spec = specs[decision_key]
        assert isinstance(spec.scoring, EnumScoring)
        classification = classify_enum(raw_decision, spec.scoring.values)
        if classification.is_valid:
            normalized = classification.canonical

    # Step 6: map canonical enum value → Role
    winner = _ENUM_TO_ROLE.get(normalized) if normalized is not None else None
    is_tie = normalized is not None and _ENUM_TO_ROLE.get(normalized) is None

    return VerdictResult(
        winner=winner if not is_tie else None,
        reason=reason or text,
        raw_decision=raw_decision,
        normalized_decision=normalized,
    )


# ============================================================================
# 3. Test cases
# ============================================================================


def _sep(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def test_field_resolution():
    """Show that _resolve_fields produces the right FieldSpecs from MDL YAML."""
    _sep("Field Resolution")

    current_specs = _resolve_fields(CURRENT_YAML_FIELDS)
    mdl_specs = _resolve_fields(MDL_YAML_FIELDS)

    print("Current YAML → FieldSpecs:")
    for name, spec in current_specs.items():
        print(f"  {name}: type={spec.type.__name__}, scoring={spec.scoring}, normalizer={spec.normalizer}")

    print("\nMDL YAML → FieldSpecs:")
    for name, spec in mdl_specs.items():
        print(f"  {name}: type={spec.type.__name__}, scoring={spec.scoring}, normalizer={spec.normalizer}")

    # Key difference: MDL has EnumScoring on decision
    assert mdl_specs["decision"].scoring is not None
    assert isinstance(mdl_specs["decision"].scoring, EnumScoring)
    assert mdl_specs["decision"].normalizer is not None
    print("\n  decision field has EnumScoring with normalizer")


def test_field_discovery():
    """Show that we can discover decision/reasoning fields without hardcoding names."""
    _sep("Field Discovery")

    specs = _resolve_fields(MDL_YAML_FIELDS)
    decision_key, reasoning_key = discover_verdict_fields(specs)
    print(f"  decision field: {decision_key!r}")
    print(f"  reasoning field: {reasoning_key!r}")

    assert decision_key == "decision"
    assert reasoning_key == "reasoning"

    # Works even if field names change!
    renamed_yaml = {
        "explanation": {"type": "str"},
        "verdict": {
            "type": "str",
            "scoring": {"mode": "enum", "values": ["A", "B", "tie"]},
        },
    }
    specs2 = _resolve_fields(renamed_yaml)
    d2, r2 = discover_verdict_fields(specs2)
    print(f"\n  Renamed fields → decision={d2!r}, reasoning={r2!r}")
    assert d2 == "verdict"
    assert r2 == "explanation"


def test_verdict_parsing():
    """Test against all the cases that currently break."""
    _sep("Verdict Parsing (Bug 1 + Bug 2)")

    specs = _resolve_fields(MDL_YAML_FIELDS)

    cases = [
        # (description, judge_response, expected_winner, expected_normalized)
        (
            "Judge says 'A' (current code: TIE, MDL: DEBATER_A)",
            "<reasoning>Expert A gave better evidence</reasoning><decision>A</decision>",
            Role.DEBATER_A,
            "A",
        ),
        (
            "Judge says 'B' (current code: TIE, MDL: DEBATER_B)",
            "<reasoning>Expert B was more rigorous</reasoning><decision>B</decision>",
            Role.DEBATER_B,
            "B",
        ),
        (
            "Judge says 'tie' (current code: TIE, MDL: TIE)",
            "<reasoning>Both equally strong</reasoning><decision>tie</decision>",
            None,
            "tie",
        ),
        (
            "Judge says 'a' lowercase (current code: TIE, MDL: DEBATER_A)",
            "<reasoning>A wins</reasoning><decision>a</decision>",
            Role.DEBATER_A,
            "A",
        ),
        (
            "Judge says 'Tie' mixed case (current code: TIE, MDL: TIE)",
            "<reasoning>Draw</reasoning><decision>Tie</decision>",
            None,
            "tie",
        ),
        (
            "Field named 'reasoning' not 'reason' (Bug 2 — reasoning text recovered)",
            "<reasoning>This is the judge's detailed analysis</reasoning><decision>A</decision>",
            Role.DEBATER_A,
            "A",
        ),
    ]

    all_passed = True
    for desc, response, expected_winner, expected_norm in cases:
        result = parse_verdict_from_specs(response, specs)
        ok = result.winner == expected_winner and result.normalized_decision == expected_norm
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(f"  [{status}] {desc}")
        if not ok:
            print(f"         got winner={result.winner}, norm={result.normalized_decision}")
        # For Bug 2 case, verify reasoning text is captured
        if "reasoning text recovered" in desc:
            has_reason = "detailed analysis" in result.reason
            print(f"         reasoning captured: {has_reason}")
            if not has_reason:
                all_passed = False

    assert all_passed, "Some verdict parsing tests failed"


def test_think_stripping():
    """Test single canonical strip_think handles all cases (Bug 3)."""
    _sep("Think Stripping (Bug 3)")

    specs = _resolve_fields(MDL_YAML_FIELDS)

    cases = [
        (
            "Closed <thinking> tag",
            "<thinking>internal reasoning here</thinking><reasoning>A is better</reasoning><decision>A</decision>",
            Role.DEBATER_A,
        ),
        (
            "Closed <think> tag",
            "<think>internal reasoning</think><reasoning>B wins</reasoning><decision>B</decision>",
            Role.DEBATER_B,
        ),
        (
            "Unclosed <thinking> tag (truncated by token limit)",
            "<thinking>this reasoning was cut off by max_tok",
            # No fields extractable after stripping — should gracefully return no winner
            None,
        ),
        (
            "Unclosed <think> tag (truncated)",
            "<think>cut off reasoning without closing tag",
            None,
        ),
        (
            "Thinking + valid verdict after",
            "<thinking>Let me think about this carefully...</thinking>\n<reasoning>A provided better evidence</reasoning>\n<decision>A</decision>",
            Role.DEBATER_A,
        ),
    ]

    all_passed = True
    for desc, response, expected_winner in cases:
        result = parse_verdict_from_specs(response, specs)
        ok = result.winner == expected_winner
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(f"  [{status}] {desc}")
        if not ok:
            print(f"         got winner={result.winner}")

    assert all_passed, "Some think-stripping tests failed"


def test_current_code_breaks():
    """Demonstrate that the CURRENT code breaks on these inputs."""
    _sep("Current Code Failure Modes (for reference)")

    from tinker_cookbook.recipes.multiplayer_rl.debate.scoring.judge import _parse_verdict, _VALID_DECISIONS

    # Bug 1: "A" not in _VALID_DECISIONS
    print(f"  _VALID_DECISIONS keys: {list(_VALID_DECISIONS.keys())}")
    print(f"  'a' in _VALID_DECISIONS: {'a' in _VALID_DECISIONS}")
    print(f"  'debater_a' in _VALID_DECISIONS: {'debater_a' in _VALID_DECISIONS}")

    # Parse a judge response that says "A"
    fields = {"decision": "A", "reasoning": "Expert A was better"}
    outcome = _parse_verdict("...", fields)
    print(f"\n  Judge says 'A' → winner={outcome.winner} (should be DEBATER_A)")
    assert outcome.winner is None, "Expected None (bug), got a winner — bug may be fixed already?"

    # Bug 2: "reasoning" vs "reason"
    print(f"\n  fields has 'reasoning': {'reasoning' in fields}")
    print(f"  _parse_verdict reads 'reason': fields.get('reason')={fields.get('reason')!r}")
    print(f"  _parse_verdict reads 'reasoning': fields.get('reasoning')={fields.get('reasoning')!r}")
    print(f"  verdict_text will be empty because it reads 'reason', not 'reasoning'")


def test_enum_normalizer_robustness():
    """Show classify_enum handles edge cases the current hardcoded dict can't."""
    _sep("EnumScoring Normalization Robustness")

    values = ("A", "B", "tie")
    cases = [
        ("A", "A", True),
        ("a", "A", True),
        ("B", "B", True),
        ("b", "B", True),
        ("tie", "tie", True),
        ("Tie", "tie", True),
        ("TIE", "tie", True),
        ("debater_a", None, False),  # correctly rejects old format
        ("debater_b", None, False),
        ("C", None, False),  # correctly rejects invalid
        ("", None, False),
        ("A B", None, False),  # correctly rejects ambiguous
    ]

    all_passed = True
    for raw, expected_canonical, expected_valid in cases:
        result = classify_enum(raw, values)
        ok = result.canonical == expected_canonical and result.is_valid == expected_valid
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(f"  [{status}] {raw!r:15s} → canonical={result.canonical!r}, valid={result.is_valid}")

    assert all_passed, "Some enum normalization tests failed"


if __name__ == "__main__":
    test_field_resolution()
    test_field_discovery()
    test_verdict_parsing()
    test_think_stripping()
    test_current_code_breaks()
    test_enum_normalizer_robustness()
    print(f"\n{'='*60}")
    print("All tests passed.")
    print(f"{'='*60}")
