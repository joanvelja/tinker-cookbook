"""Train script for debate RL with frozen opponent and LLM judge."""

from __future__ import annotations

import asyncio
import random
from datetime import datetime

import chz
import tinker
from datasets import load_dataset
from tinker.lib.public_interfaces.service_client import RetryConfig

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.usage import UsageTracker

from ..env import DebateDataset, DebateProblem
from ..eval.dataset_adapter import GPQAAdapter
from ..eval.evaluator import DebateInspectEvaluatorBuilder
from ..scoring.judge import LLMJudgeCallback, zero_sum_outcome_reward
from ..types import ProtocolKind


def _recommended_max_connections(batch_size: int, group_size: int) -> int:
    # For the user's target config batch_size=32, group_size=8 this yields 256.
    return max(16, batch_size * group_size)


def _load_gpqa(
    subset: str = "gpqa_diamond",
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    ds = load_dataset("Idavidrein/gpqa", subset, split="train").shuffle(seed=seed)
    rows = [ds[i] for i in range(len(ds))]
    n_test = max(1, int(len(rows) * test_fraction))
    return rows[:-n_test], rows[-n_test:]


def _gpqa_to_problems(rows: list[dict], seed: int = 42) -> list[DebateProblem]:
    """Convert GPQA rows to free-debate problems with shuffled MCQ options.

    Returns (task_prompt, "", "", target_label) tuples — empty answer strings
    for free debate, with target_label tracking the ground truth for accuracy.
    """
    rng = random.Random(seed)
    problems: list[DebateProblem] = []
    for row in rows:
        correct = row["Correct Answer"]
        wrong = [row[f"Incorrect Answer {i}"] for i in (1, 2, 3)]

        # Shuffle into ABCD, track correct label.
        options = [correct] + wrong
        rng.shuffle(options)
        target_label = chr(ord("A") + options.index(correct))

        # Format as MCQ prompt.
        option_lines = "\n".join(f"{chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options))
        task_prompt = f"{row['Question']}\n\n{option_lines}"
        problems.append((task_prompt, "", "", target_label))
    return problems


@chz.chz
class DebateRLDatasetBuilder(RLDatasetBuilder):
    """Builds debate RL datasets with opponent completer and LLM judge."""

    model_name: str
    renderer_name: str | None = None
    reasoning_effort: str | None = None
    opponent_model: str
    judge_model: str
    opponent_max_tokens: int = 8192
    judge_max_tokens: int = 4096
    gpqa_subset: str = "gpqa_extended"
    protocol_kind: ProtocolKind = ProtocolKind.SEQUENTIAL
    num_rounds: int = 2
    batch_size: int = 32
    group_size: int = 8
    randomize_position: bool = True
    prompts_ref: str = "judge_exploit"
    open_reasoning: bool = False
    base_url: str | None = None
    episode_log_dir: str | None = None
    max_connections: int = 256
    progress_timeout_s: int = 900

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        # Per-role renderer construction: each actor gets its own model-appropriate renderer.
        trained_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name, reasoning_effort=self.reasoning_effort
        )
        trained_renderer = get_renderer(trained_name, get_tokenizer(self.model_name))

        opp_name = model_info.get_recommended_renderer_name(
            self.opponent_model, reasoning_effort=self.reasoning_effort
        )
        opponent_renderer = get_renderer(opp_name, get_tokenizer(self.opponent_model))

        judge_name = model_info.get_recommended_renderer_name(
            self.judge_model, reasoning_effort=self.reasoning_effort
        )
        judge_renderer = get_renderer(judge_name, get_tokenizer(self.judge_model))

        tracker: UsageTracker | None = getattr(self, "_usage_tracker", None)
        retry_config = RetryConfig(
            max_connections=self.max_connections,
            progress_timeout=self.progress_timeout_s,
        )
        service = tinker.ServiceClient(base_url=self.base_url)
        opponent_sampling = service.create_sampling_client(
            base_model=self.opponent_model,
            retry_config=retry_config,
        )
        judge_sampling = service.create_sampling_client(
            base_model=self.judge_model,
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
        judge_completer = TinkerMessageCompleter(
            sampling_client=judge_sampling,
            renderer=judge_renderer,
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
            renderer=trained_renderer,
            protocol_kind=self.protocol_kind,
            num_rounds=self.num_rounds,
            open_reasoning=self.open_reasoning,
            judge_callback=judge_callback,
            outcome_reward_fn=zero_sum_outcome_reward,
            opponent_completer=opponent_completer,
            opponent_renderer=opponent_renderer,
            group_size=self.group_size,
            randomize_position=self.randomize_position,
            prompts_ref=self.prompts_ref,
            episode_log_dir=self.episode_log_dir,
        )

        test_ds: RLDataset | None = None
        if test_problems:
            test_ds = DebateDataset(
                problems=test_problems,
                batch_size=len(test_problems),
                renderer=trained_renderer,
                protocol_kind=self.protocol_kind,
                num_rounds=self.num_rounds,
                open_reasoning=self.open_reasoning,
                judge_callback=judge_callback,
                outcome_reward_fn=zero_sum_outcome_reward,
                opponent_completer=opponent_completer,
                opponent_renderer=opponent_renderer,
                group_size=self.group_size,
                randomize_position=self.randomize_position,
                prompts_ref=self.prompts_ref,
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
    gpqa_subset: str = "gpqa_extended"
    protocol_kind: ProtocolKind = ProtocolKind.SEQUENTIAL
    num_rounds: int = 2
    batch_size: int = 32
    group_size: int = 8
    learning_rate: float = 3e-5
    max_tokens: int = 8192
    randomize_position: bool = True
    prompts_ref: str = "judge_exploit"
    open_reasoning: bool = False
    kl_penalty_coef: float = 0.0
    inspect_eval: DebateInspectEvaluatorBuilder | None = None
    eval_every: int = 10
    save_every: int = 20
    wandb_project: str | None = "debate-judge-exploitation"
    wandb_name: str | None = None
    log_path: str | None = None
    episode_log_dir: str | None = None
    base_url: str | None = None
    max_connections: int | None = None
    progress_timeout_s: int = 900


def build_config(cli: CLIConfig) -> train.Config:
    model_name = cli.model_name
    max_connections = cli.max_connections or _recommended_max_connections(
        cli.batch_size, cli.group_size
    )

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
        gpqa_subset=cli.gpqa_subset,
        protocol_kind=cli.protocol_kind,
        num_rounds=cli.num_rounds,
        batch_size=cli.batch_size,
        group_size=cli.group_size,
        randomize_position=cli.randomize_position,
        prompts_ref=cli.prompts_ref,
        open_reasoning=cli.open_reasoning,
        base_url=cli.base_url,
        episode_log_dir=cli.episode_log_dir,
        max_connections=max_connections,
        progress_timeout_s=cli.progress_timeout_s,
    )

    # Resolve renderer name for eval builder (eval still takes a concrete name).
    eval_renderer_name = cli.renderer_name or model_info.get_recommended_renderer_name(
        model_name, reasoning_effort=cli.reasoning_effort
    )

    if cli.inspect_eval is not None:
        inspect_builder = chz.replace(
            cli.inspect_eval,
            renderer_name=cli.inspect_eval.renderer_name or eval_renderer_name,
            model_name=cli.inspect_eval.model_name or model_name,
            reasoning_effort=cli.reasoning_effort,
            max_connections=max_connections,
            progress_timeout_s=cli.progress_timeout_s,
        )
    else:
        # Default: GPQA eval matching training config.
        inspect_builder = DebateInspectEvaluatorBuilder(
            adapter=GPQAAdapter(free_debate=True, limit=10),
            prompts_ref=cli.prompts_ref,
            num_rounds=cli.num_rounds,
            protocol_kind=cli.protocol_kind,
            open_reasoning=cli.open_reasoning,
            randomize_position=cli.randomize_position,
            opponent_model=cli.opponent_model,
            judge_model=cli.judge_model,
            opponent_max_tokens=cli.opponent_max_tokens,
            judge_max_tokens=cli.judge_max_tokens,
            renderer_name=eval_renderer_name,
            reasoning_effort=cli.reasoning_effort,
            model_name=model_name,
            base_url=cli.base_url,
            max_connections=max_connections,
            progress_timeout_s=cli.progress_timeout_s,
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
        save_every=cli.save_every,
        evaluator_builders=evaluator_builders,
        wandb_project=cli.wandb_project,
        wandb_name=wandb_name,
        base_url=cli.base_url,
        sampling_max_connections=max_connections,
        sampling_progress_timeout=cli.progress_timeout_s,
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
