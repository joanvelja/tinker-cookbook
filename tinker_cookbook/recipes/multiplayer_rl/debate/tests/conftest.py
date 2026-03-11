"""Shared test fixtures for the debate recipe."""

from __future__ import annotations

from typing import Any

from tinker_cookbook.recipes.multiplayer_rl.debate.core.schedule import build_schedule
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateProblemSpec,
    DebateSpec,
    ProtocolKind,
    Role,
    ScoringMode,
    TurnSlot,
)

_UNSET: Any = object()


def make_spec(
    *,
    debate_id: str = "test",
    task_prompt: str = "What is X?\nA) Foo\nB) Bar\nC) Baz\nD) Qux",
    scoring_mode: ScoringMode = ScoringMode.MCQ,
    answer_by_role: dict[Role, str] | None = _UNSET,
    target: str | None = None,
    num_rounds: int = 2,
    schedule: tuple[TurnSlot, ...] | None = None,
    open_reasoning: bool = False,
    protocol_kind: ProtocolKind = ProtocolKind.SEQUENTIAL,
    prompts_ref: str = "default",
) -> DebateSpec:
    """Shared factory for DebateSpec in tests.

    If ``schedule`` is provided it is used directly; otherwise one is built
    from ``num_rounds`` via ``build_schedule(protocol_kind, num_rounds)``.

    ``answer_by_role`` defaults to ``{DEBATER_A: "A", DEBATER_B: "B"}``.
    Pass ``None`` explicitly to create a no-stance spec.
    """
    if schedule is None:
        schedule = build_schedule(protocol_kind, num_rounds)
    if answer_by_role is _UNSET:
        answer_by_role = {Role.DEBATER_A: "A", Role.DEBATER_B: "B"}

    return DebateSpec(
        debate_id=debate_id,
        problem=DebateProblemSpec(
            task_prompt=task_prompt,
            scoring_mode=scoring_mode,
            answer_by_role=answer_by_role,
            target=target,
        ),
        schedule=schedule,
        open_reasoning=open_reasoning,
        protocol_kind=protocol_kind,
        prompts_ref=prompts_ref,
    )
