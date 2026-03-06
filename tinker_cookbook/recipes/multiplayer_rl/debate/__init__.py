"""Debate environment for Tinker RL."""

from .env import (
    DebateBranchGroupBuilder,
    DebateDataset,
    DebateDatasetBuilder,
    DebateEnv,
    DebateGroupBuilder,
    DebateProblem,
)
from .scoring.judge import LLMJudgeCallback, zero_sum_outcome_reward
from .scoring.metrics import MetricFn, MetricResult, mcq_debate_metrics
from .scoring.providers import DebateScorerBuilder
from .plugins import JudgeCallback, OutcomeRewardFn, StepRewardFn
from .prompts import DebatePrompts, resolve_prompts
from .scoring.fields import FieldSpec
from .core.runtime import DebateRuntime, SubmitResult
from .core.schedule import build_schedule
from .types import (
    ActionResult,
    DebateOutcome,
    DebateSnapshot,
    DebateSpec,
    DebateState,
    JudgeDecision,
    JudgeRequest,
    Phase,
    ProtocolKind,
    Role,
    ScoringMode,
    TurnSlot,
    VisibilityPolicy,
    TurnTicket,
    Utterance,
)
from .core.visibility import build_generation_messages, get_visible_messages

__all__ = [
    # Enums
    "Phase",
    "ProtocolKind",
    "Role",
    "ScoringMode",
    "VisibilityPolicy",
    # Frozen dataclasses
    "ActionResult",
    "DebateOutcome",
    "DebateSnapshot",
    "DebateSpec",
    "DebateState",
    "JudgeDecision",
    "JudgeRequest",
    "TurnSlot",
    "TurnTicket",
    "Utterance",
    # Protocols
    "JudgeCallback",
    "OutcomeRewardFn",
    "StepRewardFn",
    # Env + builders
    "DebateEnv",
    "DebateGroupBuilder",
    "DebateBranchGroupBuilder",
    "DebateDataset",
    "DebateDatasetBuilder",
    "DebateProblem",
    # Metrics
    "MetricFn",
    "MetricResult",
    "mcq_debate_metrics",
    # Runtime
    "DebateRuntime",
    "SubmitResult",
    # Concrete implementations
    "LLMJudgeCallback",
    "zero_sum_outcome_reward",
    "DebateScorerBuilder",
    # Prompts
    "DebatePrompts",
    "FieldSpec",
    "resolve_prompts",
    # Functions
    "build_schedule",
    "build_generation_messages",
    "get_visible_messages",
]
