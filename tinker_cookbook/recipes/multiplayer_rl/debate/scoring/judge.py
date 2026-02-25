"""LLM-based judge for debate evaluation."""

from __future__ import annotations

import re
from typing import Any

from tinker_cookbook.completers import MessageCompleter
from tinker_cookbook.renderers import Message

from .parsing import extract_fields
from ..prompts import resolve_prompts
from ..types import DebateOutcome, JudgeDecision, JudgeRequest, Role
from ..core.visibility import build_generation_messages

_XML_TAG_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)
_VALID_DECISIONS = {"debater_a": Role.DEBATER_A, "debater_b": Role.DEBATER_B}
_TIE_SCORES: dict[Role, float] = {Role.DEBATER_A: 0.0, Role.DEBATER_B: 0.0}

class LLMJudgeCallback:
    """JudgeCallback that uses a frozen LLM to evaluate debates."""

    def __init__(self, judge_completer: MessageCompleter) -> None:
        self._completer = judge_completer

    async def on_boundary(self, request: JudgeRequest) -> JudgeDecision | None:
        return None

    async def on_final(self, request: JudgeRequest) -> DebateOutcome:
        state = request.state
        messages, _prefill = build_generation_messages(state, Role.JUDGE, trigger="final")
        response = await self._completer(messages)
        text = response.get("content", "") or ""

        # Schema-driven extraction if prompts define judge fields
        prompts = resolve_prompts(state.spec.prompts_ref)
        specs = prompts.get_field_specs("judge", "final")
        if specs:
            fields = extract_fields(text, specs)
        else:
            fields = None

        return _parse_verdict(text, fields)


def _extract_xml_fields(text: str) -> dict[str, str]:
    """Extract all <tag>value</tag> pairs from text."""
    return {m.group(1): m.group(2).strip() for m in _XML_TAG_RE.finditer(text)}


def _parse_verdict(text: str, fields: dict[str, Any] | None = None) -> DebateOutcome:
    """Parse verdict from judge response. Uses schema-extracted fields if available, else regex fallback."""
    if fields is None:
        fields = _extract_xml_fields(text)
    elif "decision" not in fields:
        # Schema extraction succeeded but missed decision — merge with regex fallback.
        fallback = _extract_xml_fields(text)
        fallback.update(fields)
        fields = fallback

    decision_raw = str(fields.get("decision", "")).strip().lower()
    reason = str(fields.get("reason", ""))

    winner = _VALID_DECISIONS.get(decision_raw)
    if winner is None:
        return DebateOutcome(winner=None, scores_by_role=_TIE_SCORES, verdict_text=reason or text)

    loser = Role.DEBATER_B if winner == Role.DEBATER_A else Role.DEBATER_A
    return DebateOutcome(
        winner=winner,
        scores_by_role={winner: 1.0, loser: -1.0},
        verdict_text=reason,
    )


def zero_sum_outcome_reward(outcome: DebateOutcome) -> dict[Role, float]:
    """Map outcome to rewards: winner +1, loser -1, tie 0/0."""
    if outcome.winner is None:
        return {Role.DEBATER_A: 0.0, Role.DEBATER_B: 0.0}
    loser = Role.DEBATER_B if outcome.winner == Role.DEBATER_A else Role.DEBATER_A
    return {outcome.winner: 1.0, loser: -1.0}
