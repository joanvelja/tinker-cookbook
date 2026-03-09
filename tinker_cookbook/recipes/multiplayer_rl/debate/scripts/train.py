"""Train script for debate RL with frozen opponent and LLM judge."""

from __future__ import annotations

import asyncio
from datetime import datetime

import chz
import tinker
from tinker.lib.public_interfaces.service_client import RetryConfig

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl import train
from tinker_cookbook.rl.train import StreamMinibatchConfig
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.usage import UsageTracker

from ..dataset import DebateDataset
from ..eval.dataset_adapter import ProblemsAdapter
from ..eval.evaluator import DebateInspectEvaluatorBuilder
from ..scoring.judge import LLMJudgeCallback, zero_sum_outcome_reward
from ..scoring.providers import DebateScorerBuilder
from ..sources import GPQAProblemSource, ProblemSource
from ..types import DebateGameSpec, DebateProblemSpec, ProtocolKind, ScoringMode


def _recommended_max_connections(batch_size: int, group_size: int, self_play: bool = False) -> int:
    # For the user's target config batch_size=32, group_size=8 this yields 256.
    # Self-play has 2 trained agents per game, so double the connections.
    agents_per_game = 2 if self_play else 1
    return max(16, batch_size * group_size * agents_per_game)


@chz.chz
class DebateRLDatasetBuilder(RLDatasetBuilder):
    """Builds debate RL datasets with opponent completer and LLM judge.

    Receives pre-loaded problems from ``build_config``. This separation
    ensures the eval adapter can draw from the *same* test split,
    eliminating train/test contamination from independent HF re-loads.
    """

    model_name: str
    renderer_name: str | None = None
    reasoning_effort: str | None = None
    opponent_model: str
    judge_model: str
    opponent_max_tokens: int = 8192
    judge_max_tokens: int = 4096
    train_problems: list[DebateProblemSpec] = chz.field(default_factory=list)
    test_problems: list[DebateProblemSpec] = chz.field(default_factory=list)
    protocol_kind: ProtocolKind = ProtocolKind.SEQUENTIAL
    num_rounds: int = 2
    batch_size: int = 32
    group_size: int = 8
    randomize_position: bool = True
    prompts_ref: str = "judge_exploit"
    open_reasoning: bool = False
    self_play: bool = False
    base_url: str | None = None
    episode_log_dir: str | None = None
    max_connections: int = 256
    progress_timeout_s: int = 900
    scorer_builder: DebateScorerBuilder | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        trained_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name, reasoning_effort=self.reasoning_effort
        )
        trained_renderer = get_renderer(trained_name, get_tokenizer(self.model_name))

        tracker: UsageTracker | None = getattr(self, "_usage_tracker", None)
        retry_config = RetryConfig(
            max_connections=self.max_connections,
            progress_timeout=self.progress_timeout_s,
        )
        service = tinker.ServiceClient(base_url=self.base_url)

        if not self.self_play:
            opp_name = model_info.get_recommended_renderer_name(
                self.opponent_model, reasoning_effort=self.reasoning_effort
            )
            opponent_renderer = get_renderer(opp_name, get_tokenizer(self.opponent_model))
            opponent_sampling = service.create_sampling_client(
                base_model=self.opponent_model,
                retry_config=retry_config,
            )
            opponent_completer = TinkerMessageCompleter(
                sampling_client=opponent_sampling,
                renderer=opponent_renderer,
                max_tokens=self.opponent_max_tokens,
                usage_tracker=tracker,
                actor="opponent",
                model_name=self.opponent_model,
            )
        else:
            opponent_completer = None
            opponent_renderer = None

        judge_name = model_info.get_recommended_renderer_name(
            self.judge_model, reasoning_effort=self.reasoning_effort
        )
        judge_renderer = get_renderer(judge_name, get_tokenizer(self.judge_model))
        judge_sampling = service.create_sampling_client(
            base_model=self.judge_model,
            retry_config=retry_config,
        )
        judge_completer = TinkerMessageCompleter(
            sampling_client=judge_sampling,
            renderer=judge_renderer,
            max_tokens=self.judge_max_tokens,
            usage_tracker=tracker,
            actor="judge",
            model_name=self.judge_model,
        )
        judge_callback = LLMJudgeCallback(judge_completer)
        scorer_client = (
            self.scorer_builder.build(usage_tracker=tracker)
            if self.scorer_builder is not None
            else None
        )
        scorer_parallelism = (
            self.scorer_builder.max_connections
            if self.scorer_builder is not None
            else self.max_connections
        )

        game = DebateGameSpec(
            protocol_kind=self.protocol_kind,
            num_rounds=self.num_rounds,
            prompts_ref=self.prompts_ref,
            open_reasoning=self.open_reasoning,
        )

        train_ds = DebateDataset(
            problems=self.train_problems,
            batch_size=self.batch_size,
            group_size=self.group_size,
            game=game,
            renderer=trained_renderer,
            judge_callback=judge_callback,
            outcome_reward_fn=zero_sum_outcome_reward,
            opponent_completer=opponent_completer,
            opponent_renderer=opponent_renderer,
            randomize_position=self.randomize_position,
            scorer=scorer_client,
            scorer_parallelism=scorer_parallelism,
            episode_log_dir=self.episode_log_dir,
        )

        test_ds: RLDataset | None = None
        if self.test_problems:
            test_ds = DebateDataset(
                problems=self.test_problems,
                batch_size=len(self.test_problems),
                group_size=self.group_size,
                game=game,
                renderer=trained_renderer,
                judge_callback=judge_callback,
                outcome_reward_fn=zero_sum_outcome_reward,
                opponent_completer=opponent_completer,
                opponent_renderer=opponent_renderer,
                randomize_position=self.randomize_position,
                scorer=scorer_client,
                scorer_parallelism=scorer_parallelism,
                episode_log_dir=self.episode_log_dir,
            )

        return train_ds, test_ds


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name: str | None = None
    reasoning_effort: str | None = None
    opponent_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    judge_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    opponent_max_tokens: int = 8192
    judge_max_tokens: int = 4096
    problem_source: ProblemSource = GPQAProblemSource()
    protocol_kind: ProtocolKind = ProtocolKind.SEQUENTIAL
    num_rounds: int = 2
    batch_size: int = 32
    group_size: int = 8
    learning_rate: float = 3e-5
    max_tokens: int = 8192
    randomize_position: bool = True
    self_play: bool = False
    prompts_ref: str | None = None
    open_reasoning: bool = False
    kl_penalty_coef: float = 0.0
    inspect_eval: DebateInspectEvaluatorBuilder | None = None
    eval_limit: int | None = 25
    eval_every: int = 10
    eval_on_start: bool = True
    save_every: int = 20
    wandb_project: str | None = "debate-judge-exploitation"
    wandb_name: str | None = None
    log_path: str | None = None
    episode_log_dir: str | None = None
    base_url: str | None = None
    max_connections: int | None = None
    progress_timeout_s: int = 900
    scorer_builder: DebateScorerBuilder | None = None
    num_minibatches: int | None = None  # None = sync training (default)


def build_config(cli: CLIConfig) -> train.Config:
    if cli.num_minibatches is not None and cli.batch_size % cli.num_minibatches != 0:
        raise ValueError(
            f"batch_size ({cli.batch_size}) must be divisible by "
            f"num_minibatches ({cli.num_minibatches})"
        )

    scoring_mode = cli.problem_source.scoring_mode()
    if scoring_mode == ScoringMode.OPEN_ENDED and cli.scorer_builder is None:
        raise ValueError(
            f"problem_source {type(cli.problem_source).__name__} produces "
            f"OPEN_ENDED problems, which require scorer_builder to be set"
        )

    model_name = cli.model_name
    prompts_ref = cli.prompts_ref or cli.problem_source.default_prompts_ref()
    max_connections = cli.max_connections or _recommended_max_connections(
        cli.batch_size, cli.group_size, self_play=cli.self_play
    )
    scorer_parallelism = (
        cli.scorer_builder.max_connections if cli.scorer_builder is not None else max_connections
    )

    # Self-play: both agents share the same policy, so randomize_position is meaningless.
    randomize_position = False if cli.self_play else cli.randomize_position

    # Load problems early so the eval adapter can draw from the test split,
    # eliminating train/test contamination from independent HF re-loads.
    train_problems, test_problems = cli.problem_source.load()

    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"debate-{model_name.split('/')[-1]}-"
        f"{cli.group_size}group-{cli.batch_size}batch-"
        f"{cli.learning_rate}lr-{date_and_time}"
    )
    log_path = cli.log_path or f"/tmp/tinker-examples/debate-rl/{run_name}"
    wandb_name = cli.wandb_name or run_name

    dataset_builder = DebateRLDatasetBuilder(
        model_name=model_name,
        renderer_name=cli.renderer_name,
        reasoning_effort=cli.reasoning_effort,
        opponent_model=cli.opponent_model,
        judge_model=cli.judge_model,
        opponent_max_tokens=cli.opponent_max_tokens,
        judge_max_tokens=cli.judge_max_tokens,
        train_problems=train_problems,
        test_problems=test_problems,
        protocol_kind=cli.protocol_kind,
        num_rounds=cli.num_rounds,
        batch_size=cli.batch_size,
        group_size=cli.group_size,
        randomize_position=randomize_position,
        self_play=cli.self_play,
        prompts_ref=prompts_ref,
        open_reasoning=cli.open_reasoning,
        base_url=cli.base_url,
        episode_log_dir=cli.episode_log_dir,
        max_connections=max_connections,
        progress_timeout_s=cli.progress_timeout_s,
        scorer_builder=cli.scorer_builder,
    )

    # Resolve renderer name for eval builder (eval still takes a concrete name).
    eval_renderer_name = cli.renderer_name or model_info.get_recommended_renderer_name(
        model_name, reasoning_effort=cli.reasoning_effort
    )

    # Eval parallelism: one debate per sample, 3 actors each.
    eval_n = cli.eval_limit if cli.eval_limit is not None else len(test_problems)
    eval_max_connections = max(16, eval_n * 3)

    if cli.inspect_eval is not None:
        inspect_builder = chz.replace(
            cli.inspect_eval,
            renderer_name=cli.inspect_eval.renderer_name or eval_renderer_name,
            model_name=cli.inspect_eval.model_name or model_name,
            reasoning_effort=cli.reasoning_effort,
            max_connections=eval_max_connections,
            progress_timeout_s=cli.progress_timeout_s,
            scorer_builder=cli.inspect_eval.scorer_builder or cli.scorer_builder,
            scorer_parallelism=(
                cli.inspect_eval.scorer_parallelism
                if cli.inspect_eval.scorer_parallelism is not None
                else scorer_parallelism
            ),
        )
    else:
        # Eval draws from the test split — no contamination.
        eval_adapter = ProblemsAdapter(
            test_problems,
            source="gpqa_eval",
            limit=cli.eval_limit,
        )
        # Self-play training → self-play eval (opponent_model=None triggers
        # same-weights opponent in DebateInspectEvaluator).
        inspect_builder = DebateInspectEvaluatorBuilder(
            adapter=eval_adapter,
            prompts_ref=prompts_ref,
            num_rounds=cli.num_rounds,
            protocol_kind=cli.protocol_kind,
            open_reasoning=cli.open_reasoning,
            randomize_position=randomize_position,
            opponent_model=None if cli.self_play else cli.opponent_model,
            judge_model=cli.judge_model,
            opponent_max_tokens=cli.opponent_max_tokens,
            judge_max_tokens=cli.judge_max_tokens,
            renderer_name=eval_renderer_name,
            reasoning_effort=cli.reasoning_effort,
            model_name=model_name,
            base_url=cli.base_url,
            max_connections=eval_max_connections,
            progress_timeout_s=cli.progress_timeout_s,
            scorer_builder=cli.scorer_builder,
            scorer_parallelism=scorer_parallelism,
        )
    evaluator_builders = [inspect_builder]

    return train.Config(
        model_name=model_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli.learning_rate,
        max_tokens=cli.max_tokens,
        kl_penalty_coef=cli.kl_penalty_coef,
        eval_every=cli.eval_every,
        eval_on_start=cli.eval_on_start,
        save_every=cli.save_every,
        evaluator_builders=evaluator_builders,
        wandb_project=cli.wandb_project,
        wandb_name=wandb_name,
        base_url=cli.base_url,
        sampling_max_connections=max_connections,
        sampling_progress_timeout=cli.progress_timeout_s,
        stream_minibatch_config=(
            StreamMinibatchConfig(
                groups_per_batch=cli.batch_size,
                num_minibatches=cli.num_minibatches,
            )
            if cli.num_minibatches is not None
            else None
        ),
    )


def main():
    cli_config = chz.entrypoint(CLIConfig)
    config = build_config(cli_config)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    usage_tracker = UsageTracker()
    # Inject tracker into dataset builder for opponent/judge completers.
    object.__setattr__(config.dataset_builder, "_usage_tracker", usage_tracker)
    asyncio.run(train.main(config, usage_tracker=usage_tracker))


if __name__ == "__main__":
    main()
