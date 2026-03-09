"""Frozen types for the debate environment."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal, Mapping

_THINK_RE = re.compile(r"<think(?:ing)?[^>]*>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


class Role(StrEnum):
    DEBATER_A = "debater_a"
    DEBATER_B = "debater_b"
    JUDGE = "judge"


class Phase(StrEnum):
    PROPOSE = "propose"
    CRITIQUE = "critique"
    JUDGE_QUERY = "judge_query"
    JUDGE_VERDICT = "judge_verdict"


# Prompt-lookup keys that extend Phase values.  Single source of truth —
# used by prompts/, core/runtime.py, and core/visibility.py.
PHASE_DONE: str = "done"
TRIGGER_FINAL: str = "final"
TRIGGER_BOUNDARY: str = "boundary"


class ProtocolKind(StrEnum):
    SEQUENTIAL = "sequential"
    SIMULTANEOUS = "simultaneous"
    HYBRID = "hybrid"


class ScoringMode(StrEnum):
    MCQ = "mcq"
    OPEN_ENDED = "open_ended"


class VisibilityPolicy(StrEnum):
    ALL_PRIOR = "all_prior"
    COMPLETED_ROUNDS_ONLY = "completed_rounds_only"


@dataclass(frozen=True)
class TurnSlot:
    slot_id: int
    round_index: int
    phase: Phase
    actors: tuple[Role, ...]  # len>1 = simultaneous barrier
    boundary_after: bool  # end-of-round marker
    visibility_policy: VisibilityPolicy  # key into visibility registry


@dataclass(frozen=True)
class Utterance:
    role: Role
    round_index: int
    phase: Phase
    text: str
    token_count: int
    slot_id: int
    fields: Mapping[str, Any] | None = None
    stripped_text: str = field(init=False)

    def __post_init__(self) -> None:
        if self.fields is not None:
            object.__setattr__(self, "fields", _freeze_mapping(self.fields))
        object.__setattr__(self, "stripped_text", _strip_reasoning(self.text))


def _freeze_mapping(m: Mapping) -> MappingProxyType:
    """Wrap a mapping in MappingProxyType for true immutability."""
    if isinstance(m, MappingProxyType):
        return m
    return MappingProxyType(dict(m))


@dataclass(frozen=True)
class DebateProblemSpec:
    task_prompt: str
    scoring_mode: ScoringMode
    answer_by_role: Mapping[Role, str] | None = None
    target: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.answer_by_role is not None:
            # Normalize: strip whitespace, blank → None
            cleaned = {k: v.strip() for k, v in self.answer_by_role.items() if v and v.strip()}
            abr = _freeze_mapping(cleaned) if cleaned else None
            object.__setattr__(self, "answer_by_role", abr)
        if self.metadata is not None:
            object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @staticmethod
    def from_seat_answers(
        task_prompt: str,
        answer_a: str,
        answer_b: str,
        scoring_mode: ScoringMode,
        *,
        target: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "DebateProblemSpec":
        answer_by_role = {Role.DEBATER_A: answer_a, Role.DEBATER_B: answer_b}
        return DebateProblemSpec(
            task_prompt=task_prompt,
            scoring_mode=scoring_mode,
            answer_by_role=answer_by_role,
            target=target,
            metadata=metadata,
        )


@dataclass(frozen=True)
class DebateGameSpec:
    protocol_kind: ProtocolKind
    num_rounds: int
    prompts_ref: str = "default"
    open_reasoning: bool = False
    include_judge_turns: bool = False

    def __post_init__(self) -> None:
        if self.num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1, got {self.num_rounds}")


@dataclass(frozen=True)
class DebateSpec:
    """Static config -- does not change during episode."""

    debate_id: str
    problem: DebateProblemSpec
    schedule: tuple[TurnSlot, ...]
    open_reasoning: bool
    protocol_kind: ProtocolKind = ProtocolKind.SEQUENTIAL
    prompts_ref: str = "default"


@dataclass(frozen=True)
class JudgeDecision:
    round_index: int
    verdict: str
    score_delta_by_role: Mapping[Role, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "score_delta_by_role", _freeze_mapping(self.score_delta_by_role))


@dataclass(frozen=True)
class DebateOutcome:
    winner: Role | None
    scores_by_role: Mapping[Role, float]
    verdict_text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores_by_role", _freeze_mapping(self.scores_by_role))


@dataclass(frozen=True)
class DebateState:
    """Dynamic episode state. Immutable -- transitions produce new instances."""

    spec: DebateSpec
    slot_index: int
    rounds_completed: int
    transcript: tuple[Utterance, ...]
    pending_simultaneous: Mapping[Role, Utterance]
    judge_trace: tuple[JudgeDecision, ...]
    done: bool
    outcome: DebateOutcome | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pending_simultaneous", _freeze_mapping(self.pending_simultaneous))

    @property
    def version(self) -> int:
        return len(self.transcript) + len(self.pending_simultaneous)


def current_phase(state: DebateState) -> str:
    """Current phase string — Phase value or PHASE_DONE if schedule exhausted."""
    schedule = state.spec.schedule
    if state.slot_index < len(schedule):
        return schedule[state.slot_index].phase.value
    return PHASE_DONE


@dataclass(frozen=True)
class JudgeRequest:
    state: DebateState
    trigger: Literal["boundary", "final"]


@dataclass(frozen=True)
class ActionResult:
    """Reducer output. No reward -- rewards are a separate layer."""

    new_state: DebateState
    committed: tuple[Utterance, ...]
    boundary_reached: bool
    episode_done: bool


@dataclass(frozen=True)
class DebateSnapshot:
    state: DebateState
    renderer_name: str


@dataclass(frozen=True)
class TurnTicket:
    slot_id: int
    state_version: int
    role: Role
