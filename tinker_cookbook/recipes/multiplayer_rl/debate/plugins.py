"""Plugin protocols for the debate environment."""

from typing import Mapping, Protocol

from .types import DebateOutcome, DebateState, JudgeDecision, JudgeRequest, Role, Utterance


class StepRewardFn(Protocol):
    def __call__(
        self, before: DebateState, after: DebateState, role: Role, utterance: Utterance | None
    ) -> float: ...


class JudgeCallback(Protocol):
    async def on_boundary(self, request: JudgeRequest) -> JudgeDecision | None: ...
    async def on_final(self, request: JudgeRequest) -> DebateOutcome: ...


class OutcomeRewardFn(Protocol):
    def __call__(self, outcome: DebateOutcome) -> Mapping[Role, float]: ...
