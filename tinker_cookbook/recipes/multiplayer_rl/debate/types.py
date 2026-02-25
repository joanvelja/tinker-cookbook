"""Frozen types for the debate environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal, Mapping


class Role(StrEnum):
    DEBATER_A = "debater_a"
    DEBATER_B = "debater_b"
    JUDGE = "judge"


class Phase(StrEnum):
    PROPOSE = "propose"
    CRITIQUE = "critique"
    JUDGE_QUERY = "judge_query"
    JUDGE_VERDICT = "judge_verdict"


class ProtocolKind(StrEnum):
    SEQUENTIAL = "sequential"
    SIMULTANEOUS = "simultaneous"
    HYBRID = "hybrid"


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

    def __post_init__(self) -> None:
        if self.fields is not None:
            object.__setattr__(self, "fields", _freeze_mapping(self.fields))


def _freeze_mapping(m: Mapping) -> MappingProxyType:
    """Wrap a mapping in MappingProxyType for true immutability."""
    if isinstance(m, MappingProxyType):
        return m
    return MappingProxyType(dict(m))


@dataclass(frozen=True)
class DebateSpec:
    """Static config -- does not change during episode."""

    debate_id: str
    task_prompt: str
    answer_by_role: Mapping[Role, str] | None
    schedule: tuple[TurnSlot, ...]
    open_reasoning: bool
    protocol_kind: ProtocolKind = ProtocolKind.SEQUENTIAL
    prompts_ref: str = "default"
    target: str | None = None

    def __post_init__(self) -> None:
        if self.answer_by_role is not None:
            object.__setattr__(self, "answer_by_role", _freeze_mapping(self.answer_by_role))


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
    protocol_kind: ProtocolKind
    protocol_kwargs: Mapping[str, Any]
    renderer_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_kwargs", _freeze_mapping(self.protocol_kwargs))


@dataclass(frozen=True)
class TurnTicket:
    slot_id: int
    state_version: int
    role: Role
