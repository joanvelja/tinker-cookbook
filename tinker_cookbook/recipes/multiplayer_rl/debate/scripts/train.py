"""Train script for debate RL with frozen opponent and LLM judge."""

from __future__ import annotations

import asyncio
from dataclasses import field
from datetime import datetime

import chz
import tinker
from datasets import load_dataset

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.completers import MessageCompleter, TinkerMessageCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.usage import UsageTracker

from ..env import DebateDataset
from ..eval.evaluator import DebateInspectEvaluatorBuilder
from ..scoring.judge import LLMJudgeCallback, zero_sum_outcome_reward
from ..types import ProtocolKind, Role


def _load_gpqa(
    subset: str = "gpqa_diamond",
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    ds = load_dataset("Idavidrein/gpqa", subset, split="train").shuffle(seed=seed)
    n_test = max(1, int(len(ds) * test_fraction))
    return list(ds[:-n_test]), list(ds[-n_test:])


def _gpqa_to_problems(rows: list[dict]) -> list[tuple[str, str, str]]:
    """Convert GPQA rows to (task_prompt, answer_a, answer_b) tuples.

    For free debate (answer_by_role=None), answer_a/answer_b are empty strings.
    The DebateGroupBuilder will construct DebateSpec with answer_by_role=None
    when the completer handles stance assignment.
    """
    return [(row["Question"], "", "") for row in rows]


@chz.chz
class DebateRLDatasetBuilder(RLDatasetBuilder):
    """Builds debate RL datasets with opponent completer and LLM judge."""

    model_name: str
    renderer_name: str
    opponent_model: str
    judge_model: str
    opponent_max_tokens: int = 1024
    judge_max_tokens: int = 512
    gpqa_subset: str = "gpqa_diamond"
    protocol_kind: ProtocolKind = ProtocolKind.SEQUENTIAL
    num_rounds: int = 2
    batch_size: int = 4
    group_size: int = 4
    randomize_position: bool = True
    prompts_ref: str = "default"
    base_url: str | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        service_client = tinker.ServiceClient(base_url=self.base_url)
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        tracker: UsageTracker | None = getattr(self, "_usage_tracker", None)

        # Opponent completer (frozen model).
        opponent_sampling = service_client.create_sampling_client(base_model=self.opponent_model)
        opponent_completer = TinkerMessageCompleter(
            sampling_client=opponent_sampling,
            renderer=renderer,
            max_tokens=self.opponent_max_tokens,
            usage_tracker=tracker,
            actor="opponent",
            model_name=self.opponent_model,
        )

        # Judge completer (frozen model).
        judge_sampling = service_client.create_sampling_client(base_model=self.judge_model)
        judge_completer = TinkerMessageCompleter(
            sampling_client=judge_sampling,
            renderer=renderer,
            max_tokens=self.judge_max_tokens,
            usage_tracker=tracker,
            actor="judge",
            model_name=self.judge_model,
        )
        judge_callback = LLMJudgeCallback(judge_completer)

        # Load GPQA.
        train_rows, test_rows = _load_gpqa(self.gpqa_subset)
        train_problems = _gpqa_to_problems(train_rows)
        test_problems = _gpqa_to_problems(test_rows)

        train_ds = DebateDataset(
            problems=train_problems,
            batch_size=self.batch_size,
            renderer=renderer,
            protocol_kind=self.protocol_kind,
            num_rounds=self.num_rounds,
            judge_callback=judge_callback,
            outcome_reward_fn=zero_sum_outcome_reward,
            opponent_completer=opponent_completer,
            group_size=self.group_size,
            randomize_position=self.randomize_position,
            prompts_ref=self.prompts_ref,
        )

        test_ds: RLDataset | None = None
        if test_problems:
            test_ds = DebateDataset(
                problems=test_problems,
                batch_size=len(test_problems),
                renderer=renderer,
                protocol_kind=self.protocol_kind,
                num_rounds=self.num_rounds,
                judge_callback=judge_callback,
                outcome_reward_fn=zero_sum_outcome_reward,
                opponent_completer=opponent_completer,
                group_size=self.group_size,
                randomize_position=self.randomize_position,
                prompts_ref=self.prompts_ref,
            )

        return train_ds, test_ds


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name: str | None = None
    opponent_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    judge_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    opponent_max_tokens: int = 1024
    judge_max_tokens: int = 512
    gpqa_subset: str = "gpqa_diamond"
    protocol_kind: ProtocolKind = ProtocolKind.SEQUENTIAL
    num_rounds: int = 2
    batch_size: int = 4
    group_size: int = 4
    learning_rate: float = 3e-5
    max_tokens: int = 1024
    randomize_position: bool = True
    prompts_ref: str = "default"
    kl_penalty_coef: float = 0.0
    inspect_eval: DebateInspectEvaluatorBuilder | None = None
    eval_every: int = 5
    save_every: int = 20
    wandb_project: str | None = None
    wandb_name: str | None = None
    log_path: str | None = None
    base_url: str | None = None


def build_config(cli: CLIConfig) -> train.Config:
    model_name = cli.model_name
    renderer_name = cli.renderer_name or model_info.get_recommended_renderer_name(model_name)

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
        renderer_name=renderer_name,
        opponent_model=cli.opponent_model,
        judge_model=cli.judge_model,
        opponent_max_tokens=cli.opponent_max_tokens,
        judge_max_tokens=cli.judge_max_tokens,
        gpqa_subset=cli.gpqa_subset,
        protocol_kind=cli.protocol_kind,
        num_rounds=cli.num_rounds,
        batch_size=cli.batch_size,
        group_size=cli.group_size,
        randomize_position=cli.randomize_position,
        prompts_ref=cli.prompts_ref,
        base_url=cli.base_url,
    )

    evaluator_builders = []
    if cli.inspect_eval is not None:
        inspect_builder = chz.replace(
            cli.inspect_eval,
            renderer_name=cli.inspect_eval.renderer_name or renderer_name,
            model_name=cli.inspect_eval.model_name or model_name,
        )
        evaluator_builders.append(inspect_builder)

    return train.Config(
        model_name=model_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli.learning_rate,
        max_tokens=cli.max_tokens,
        kl_penalty_coef=cli.kl_penalty_coef,
        eval_every=cli.eval_every,
        save_every=cli.save_every,
        evaluator_builders=evaluator_builders,
        wandb_project=cli.wandb_project,
        wandb_name=wandb_name,
        base_url=cli.base_url,
    )


def main():
    cli_config = chz.entrypoint(CLIConfig)
    config = build_config(cli_config)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    usage_tracker = UsageTracker()
    # Inject tracker into dataset builder for opponent/judge completers.
    config.dataset_builder._usage_tracker = usage_tracker  # type: ignore[attr-defined]
    asyncio.run(train.main(config, usage_tracker=usage_tracker))


if __name__ == "__main__":
    main()
