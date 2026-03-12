"""LLM-based judge for debate evaluation."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from tinker_cookbook.completers import MessageCompleter
from tinker_cookbook.renderers import format_content_as_string

from .fields import EnumScoring, FieldSpec, classify_enum
from ..think import strip_think
from .parsing import extract_fields
from ..prompts import resolve_prompts
from ..types import DebateOutcome, JudgeRequest, Role
from ..core.visibility import build_generation_messages

logger = logging.getLogger(__name__)

_ENUM_TO_ROLE: dict[str, Role | None] = {
    "A": Role.DEBATER_A,
    "B": Role.DEBATER_B,
    "debater_a": Role.DEBATER_A,
    "debater_b": Role.DEBATER_B,
    "tie": None,
}

_TIE_SCORES: dict[Role, float] = {Role.DEBATER_A: 0.0, Role.DEBATER_B: 0.0}

# Regex to strip punctuation for pre-classify cleanup.
_PUNCT_RE = re.compile(r"[^\w\s]")


class LLMJudgeCallback:
    """JudgeCallback that uses a frozen LLM to evaluate debates."""

    def __init__(self, judge_completer: MessageCompleter) -> None:
        self._completer = judge_completer

    async def on_boundary(self, request: JudgeRequest) -> None:
        return None

    async def on_final(self, request: JudgeRequest) -> DebateOutcome:
        state = request.state
        messages, _prefill = build_generation_messages(state, Role.JUDGE, trigger="final")
        t0 = time.monotonic()
        response = await self._completer(messages)
        self._last_judge_wall_s = time.monotonic() - t0
        text = format_content_as_string(response["content"], separator="")

        # Schema-driven extraction if prompts define judge fields
        prompts = resolve_prompts(state.spec.prompts_ref)
        specs = prompts.get_field_specs("judge", "final")
        if specs:
            cleaned, _ = strip_think(text)
            fields = extract_fields(cleaned, specs)
        else:
            fields = None

        return _parse_verdict(text, specs or {}, fields)


def _parse_verdict(
    text: str,
    specs: dict[str, FieldSpec],
    fields: dict[str, Any] | None = None,
) -> DebateOutcome:
    """Parse verdict from judge response using schema-driven enum classification.

    Discovers the decision field by finding the one with EnumScoring,
    then classifies the raw value against the enum vocabulary.
    """
    verdict_text, _ = strip_think(text)

    if fields is None:
        logger.warning("Judge fields extraction failed; defaulting to tie")
        return DebateOutcome(winner=None, scores_by_role=_TIE_SCORES, verdict_text=verdict_text)

    # Find the decision field: the one with EnumScoring
    decision_key: str | None = None
    decision_spec: FieldSpec | None = None
    for key, spec in specs.items():
        if isinstance(spec.scoring, EnumScoring):
            decision_key = key
            decision_spec = spec
            break

    if decision_key is None or decision_spec is None:
        logger.warning("No EnumScoring field in judge specs; defaulting to tie")
        return DebateOutcome(winner=None, scores_by_role=_TIE_SCORES, verdict_text=verdict_text)

    raw = fields.get(decision_key)
    if raw is None:
        logger.warning(
            "Decision field %r missing from extracted fields; defaulting to tie", decision_key
        )
        return DebateOutcome(winner=None, scores_by_role=_TIE_SCORES, verdict_text=verdict_text)

    # Strip punctuation before classification — classify_enum/_canon_enum does NOT strip punctuation
    stripped = _PUNCT_RE.sub("", str(raw)).strip()
    assert isinstance(decision_spec.scoring, EnumScoring)
    classification = classify_enum(stripped, decision_spec.scoring.values)

    if not classification.is_valid or classification.canonical is None:
        logger.warning("Could not classify decision %r; defaulting to tie", raw)
        return DebateOutcome(winner=None, scores_by_role=_TIE_SCORES, verdict_text=verdict_text)

    winner = _ENUM_TO_ROLE.get(classification.canonical)
    if winner is None:
        # Tie
        return DebateOutcome(winner=None, scores_by_role=_TIE_SCORES, verdict_text=verdict_text)

    loser = Role.DEBATER_B if winner == Role.DEBATER_A else Role.DEBATER_A
    return DebateOutcome(
        winner=winner,
        scores_by_role={winner: 1.0, loser: -1.0},
        verdict_text=verdict_text,
    )


def zero_sum_outcome_reward(outcome: DebateOutcome) -> dict[Role, float]:
    """Map outcome to rewards: winner +1, loser -1, tie 0/0."""
    if outcome.winner is None:
        return {Role.DEBATER_A: 0.0, Role.DEBATER_B: 0.0}
    loser = Role.DEBATER_B if outcome.winner == Role.DEBATER_A else Role.DEBATER_A
    return {outcome.winner: 1.0, loser: -1.0}
