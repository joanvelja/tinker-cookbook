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


def format_penalty_reward_fn(
    before: DebateState, after: DebateState, role: Role, utterance: Utterance | None
) -> float:
    """Step reward that penalizes format violations (failed field extraction).

    Returns -0.1 when utterance.fields is None (format violation), 0.0 otherwise.
    """
    if utterance is not None and utterance.fields is None:
        return -0.1
    return 0.0
