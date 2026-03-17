"""Unified RLVR training CLI.

Supports all datasets registered in ``DATASET_BUILDER_MAP``.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Literal

import chz
from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.recipes.rlvr.builders import DATASET_BUILDER_MAP
from tinker_cookbook.rl.train import AsyncConfig, Config, KLReferenceConfig, StreamMinibatchConfig, main
from tinker_cookbook.rl.types import AdvantageScheme

logger = logging.getLogger(__name__)

DatasetChoice = Literal["math", "gsm8k", "polaris", "deepmath", "gpqa_oe", "omni_math"]


@chz.chz
class CLIConfig:
    """Command-line configuration for unified RLVR training."""

    # Required (no defaults)
    model_name: str
    dataset: DatasetChoice
    batch_size: int
    group_size: int
    max_tokens: int

    # Commonly tuned
    advantage_scheme: AdvantageScheme = "maxrl"
    lora_rank: int = 32
    renderer_name: str | None = None
    learning_rate: float | None = None  # None -> get_lr(model_name)
    temperature: float = 1.0
    eval_temperature: float = 0.6
    eval_top_p: float = 0.95
    seed: int = 42
    n_batches: int | None = None
    kl_penalty_coef: float = 0.0
    format_coef: float = 0.1
    eos_coef: float = 0.1

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Checkpointing
    load_checkpoint_path: str | None = None
    save_every: int = 20
    eval_every: int = 20
    eval_on_start: bool = True

    # Advanced
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    num_substeps: int = 1
    max_steps_off_policy: int | None = None
    compute_post_kl: bool = False
    loss_fn: LossFnType = "ppo"
    loss_fn_config: dict[str, Any] | None = None
    clip_ratio_lower: float = 0.2  # PPO clip: 1 - lower (DAPO uses 0.2)
    clip_ratio_upper: float = 0.2  # PPO clip: 1 + upper (DAPO uses 0.28)
    grad_clip_norm: float = 0.3
    remove_constant_reward_groups: bool = False

    # Stream minibatch settings
    stream_groups_per_batch: int | None = None  # enables StreamMinibatchConfig when set
    stream_num_minibatches: int = 2


def derive_loss_fn_config(cli_config: CLIConfig) -> dict[str, Any] | None:
    """Derive loss_fn_config from clip ratios. Explicit config wins."""
    if cli_config.loss_fn_config is not None:
        return cli_config.loss_fn_config
    if cli_config.loss_fn in ("ppo", "cispo"):
        return {
            "clip_low_threshold": 1.0 - cli_config.clip_ratio_lower,
            "clip_high_threshold": 1.0 + cli_config.clip_ratio_upper,
        }
    return None


async def cli_main(cli_config: CLIConfig) -> None:
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    learning_rate = cli_config.learning_rate or get_lr(cli_config.model_name)
    model_slug = cli_config.model_name.replace("/", "-")
    run_name = (
        f"{cli_config.dataset}-{model_slug}-{cli_config.lora_rank}rank"
        f"-{learning_rate}lr-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/rlvr/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    sampling_max_connections = max(16, cli_config.batch_size * cli_config.group_size)

    builder_cls = DATASET_BUILDER_MAP[cli_config.dataset]
    dataset_builder = builder_cls(
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        seed=cli_config.seed,
        n_batches=cli_config.n_batches,
        format_coef=cli_config.format_coef,
        eos_coef=cli_config.eos_coef,
    )

    loss_fn_config = derive_loss_fn_config(cli_config)

    config = Config(
        learning_rate=learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        eval_temperature=cli_config.eval_temperature,
        eval_top_p=cli_config.eval_top_p,
        advantage_scheme=cli_config.advantage_scheme,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        eval_every=cli_config.eval_every,
        eval_on_start=cli_config.eval_on_start,
        save_every=cli_config.save_every,
        sampling_max_connections=sampling_max_connections,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        kl_reference_config=KLReferenceConfig(base_model=cli_config.model_name)
        if cli_config.kl_penalty_coef > 0
        else None,
        compute_post_kl=cli_config.compute_post_kl,
        num_substeps=cli_config.num_substeps,
        loss_fn=cli_config.loss_fn,
        grad_clip_norm=cli_config.grad_clip_norm,
        loss_fn_config=loss_fn_config,
        remove_constant_reward_groups=cli_config.remove_constant_reward_groups,
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.batch_size,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
        stream_minibatch_config=StreamMinibatchConfig(
            groups_per_batch=cli_config.stream_groups_per_batch,
            num_minibatches=cli_config.stream_num_minibatches,
        )
        if cli_config.stream_groups_per_batch is not None
        else None,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
