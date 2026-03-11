"""Debate dataset and dataset builder."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Sequence

import chz
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
)
from tinker_cookbook.completers import MessageCompleter
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .scoring.metrics import MetricFn
from .plugins import JudgeCallback, OutcomeRewardFn, StepRewardFn
from .types import (
    DebateGameSpec,
    DebateProblemSpec,
    ProtocolKind,
    Role,
    ScoringMode,
)
from .builders import DebateGroupBuilder

if TYPE_CHECKING:
    from tinker_cookbook.scoring import BinaryJudgeClient


class DebateDataset(RLDataset):
    """Dataset of debate problems."""

    def __init__(
        self,
        problems: Sequence[DebateProblemSpec],
        batch_size: int,
        group_size: int,
        *,
        game: DebateGameSpec,
        renderer: Renderer,
        step_reward_fn: StepRewardFn | None = None,
        judge_callback: JudgeCallback | None = None,
        outcome_reward_fn: OutcomeRewardFn | None = None,
        include_roles: tuple[Role, ...] = (Role.DEBATER_A, Role.DEBATER_B),
        opponent_completer: MessageCompleter | None = None,
        opponent_renderer: Renderer | None = None,
        randomize_position: bool = False,
        metrics: dict[str, MetricFn] | None = None,
        scorer: BinaryJudgeClient | None = None,
        episode_log_dir: str | None = None,
    ) -> None:
        # Homogeneity validation: all problems must share scoring_mode.
        if problems:
            modes = {p.scoring_mode for p in problems}
            if len(modes) > 1:
                raise ValueError(
                    f"All problems must have the same scoring_mode, got mixed: {modes}"
                )
            scoring_mode = next(iter(modes))
            if scoring_mode == ScoringMode.OPEN_ENDED and scorer is None:
                raise ValueError(
                    "OPEN_ENDED debate scoring requires a scorer client. "
                    "Pass scorer=... on DebateDataset."
                )

        self.problems = problems
        self.batch_size = batch_size
        self.group_size = group_size
        self.game = game
        self.renderer = renderer
        self.step_reward_fn = step_reward_fn
        self.judge_callback = judge_callback
        self.outcome_reward_fn = outcome_reward_fn
        self.include_roles = include_roles
        self.opponent_completer = opponent_completer
        self.opponent_renderer = opponent_renderer
        self.randomize_position = randomize_position
        self.metrics = metrics
        self.scorer = scorer
        self.episode_log_dir = episode_log_dir

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        if not self.problems:
            return []
        start = (index * self.batch_size) % len(self.problems)
        batch_problems = [
            self.problems[(start + i) % len(self.problems)] for i in range(self.batch_size)
        ]
        return [
            DebateGroupBuilder(
                problem=problem,
                game=self.game,
                renderer=self.renderer,
                step_reward_fn=self.step_reward_fn,
                judge_callback=self.judge_callback,
                outcome_reward_fn=self.outcome_reward_fn,
                include_roles=self.include_roles,
                group_size=self.group_size,
                opponent_completer=self.opponent_completer,
                opponent_renderer=self.opponent_renderer,
                randomize_position=self.randomize_position,
                metrics=self.metrics,
                scorer=self.scorer,
                episode_log_dir=self.episode_log_dir,
            )
            for problem in batch_problems
        ]

    def __len__(self) -> int:
        return (len(self.problems) + self.batch_size - 1) // self.batch_size


@chz.chz
class DebateDatasetBuilder(RLDatasetBuilder):
    """Builder for debate datasets. Configure via chz."""

    model_name: str
    renderer_name: str
    protocol_kind: ProtocolKind = ProtocolKind.SEQUENTIAL
    num_rounds: int = 2
    open_reasoning: bool = False
    include_judge_turns: bool = False
    batch_size: int = 4
    group_size: int = 1
    prompts_ref: str = "default"
    scorer: BinaryJudgeClient | None = field(default=None, repr=False)
    train_problems: list[DebateProblemSpec] = field(default_factory=list)
    test_problems: list[DebateProblemSpec] = field(default_factory=list)

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        game = DebateGameSpec(
            protocol_kind=self.protocol_kind,
            num_rounds=self.num_rounds,
            prompts_ref=self.prompts_ref,
            open_reasoning=self.open_reasoning,
            include_judge_turns=self.include_judge_turns,
        )
        train = DebateDataset(
            problems=self.train_problems,
            batch_size=self.batch_size,
            group_size=self.group_size,
            game=game,
            renderer=renderer,
            scorer=self.scorer,
        )
        test = None
        if self.test_problems:
            test = DebateDataset(
                problems=self.test_problems,
                batch_size=len(self.test_problems),
                group_size=self.group_size,
                game=game,
                renderer=renderer,
                scorer=self.scorer,
            )
        return train, test
