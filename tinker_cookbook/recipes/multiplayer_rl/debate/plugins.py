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


def completion_and_format_reward_fn(
    before: DebateState,
    after: DebateState,
    role: Role,
    utterance: Utterance | None,
    *,
    format_coef: float = 0.1,
    eos_coef: float = 0.1,
) -> float:
    """Two-signal step reward: format extraction + response completion.

    Rewards:
      format_coef * (extracted - 1):  0 if fields extracted, -format_coef if not
      eos_coef * (completed - 1):     0 if response completed within budget,
                                      -eos_coef if truncated (no answer tag found
                                      in a response that also failed extraction)

    A response that extracts fields successfully is assumed to have completed
    (the answer tag was present and parseable). A response that fails extraction
    AND has no answer-like tag in its stripped text is treated as truncated.

    Total penalty range: 0 (both OK) to -(format_coef + eos_coef) (both fail).
    """
    if utterance is None:
        return 0.0

    # Format signal: did field extraction succeed?
    extracted = utterance.fields is not None
    format_reward = format_coef * (float(extracted) - 1.0)

    # Completion signal: did the response finish within token budget?
    # If fields extracted, the response necessarily completed (answer tag present).
    # If fields failed, check if any answer-like tag is present in the text.
    if extracted:
        completed = True
    else:
        text = utterance.stripped_text or ""
        completed = "<answer>" in text.lower() or "</answer>" in text.lower()
    eos_reward = eos_coef * (float(completed) - 1.0)

    return format_reward + eos_reward


def role_has_format_failure(state: DebateState, role: Role) -> bool:
    """Check if any utterance by this role failed field extraction.

    Only considers utterances where field_specs were defined for that
    role/phase — utterances in phases without field_specs have fields=None
    legitimately and should not trigger the gate.
    """
    from .prompts import resolve_prompts

    prompts = resolve_prompts(state.spec.prompts_ref)
    for utt in state.transcript:
        if utt.role != role:
            continue
        # Only check utterances where fields were expected
        specs = prompts.get_field_specs(role.value, utt.phase.value)
        if specs and utt.fields is None:
            return True
    return False
