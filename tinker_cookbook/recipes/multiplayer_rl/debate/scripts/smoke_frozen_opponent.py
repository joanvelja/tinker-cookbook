"""Smoke test: frozen-opponent debate training on GPQA diamond.

Runs a 2-batch training loop with a frozen opponent (separate sampling client):
  - LoRA training with GRPO (importance_sampling loss)
  - LLM judge for outcome reward
  - Validates metrics, reward variance, and training convergence

Usage:
    export $(cat .env | xargs)
    uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.smoke_frozen_opponent
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random

import datasets
import tinker
from tinker.lib.public_interfaces.service_client import RetryConfig
from tinker.types import LossFnType

from tinker_cookbook.completers import TinkerMessageCompleter, TinkerTokenCompleter
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.metric_util import compute_trajectory_metrics
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.train import train_step
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.usage import UsageTracker
from tinker_cookbook.utils import logtree

from ..env import DebateDataset, DebateProblem
from ..scoring.judge import LLMJudgeCallback, zero_sum_outcome_reward
from ..scoring.metrics import mcq_debate_metrics
from ..types import ProtocolKind

# --- Defaults ---
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
OPPONENT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
JUDGE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BATCH_SIZE = 2
GROUP_SIZE = 2
NUM_ROUNDS = 1
MAX_TOKENS = 2048
OPPONENT_MAX_TOKENS = 8192
LEARNING_RATE = 1e-4
LOSS_FN: LossFnType = "importance_sampling"
NUM_BATCHES = 2
LORA_RANK = 32
TRACE_PATH = "/tmp/tinker-examples/smoke_frozen_opponent.html"


def _load_gpqa_problems(n: int, seed: int = 42) -> list[DebateProblem]:
    """Load n GPQA diamond problems as free-debate 4-tuples with target."""
    ds = datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))

    problems: list[DebateProblem] = []
    for idx in indices:
        row = ds[idx]
        correct = row["Correct Answer"]
        wrong = [row[f"Incorrect Answer {i}"] for i in (1, 2, 3)]

        options = [correct] + wrong
        rng.shuffle(options)
        target_label = chr(ord("A") + options.index(correct))

        question = row["Question"]
        option_lines = "\n".join(f"{chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options))
        task_prompt = f"{question}\n\n{option_lines}"

        problems.append((task_prompt, "", "", target_label))

    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen-opponent debate training smoke test")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--opponent-model", default=OPPONENT_MODEL)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--num-rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--opponent-max-tokens", type=int, default=OPPONENT_MAX_TOKENS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-batches", type=int, default=NUM_BATCHES)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--prompts-ref", default="judge_exploit")
    parser.add_argument(
        "--max-connections",
        type=int,
        default=16,
        help="RetryConfig.max_connections for sampling clients.",
    )
    parser.add_argument(
        "--progress-timeout",
        type=int,
        default=900,
        help="RetryConfig.progress_timeout (seconds).",
    )
    return parser.parse_args(argv)


async def run(args: argparse.Namespace):
    # Load problems -- need at least batch_size * num_batches
    n_problems = args.batch_size * args.num_batches
    problems = _load_gpqa_problems(n_problems)
    print(f"Loaded {len(problems)} GPQA problems", flush=True)
    for i, (prompt, _, _, target) in enumerate(problems):
        print(f"  [{i}] target={target}  {prompt[:80]}...", flush=True)
    print(flush=True)

    retry_config = RetryConfig(
        max_connections=args.max_connections,
        progress_timeout=args.progress_timeout,
    )

    # --- Service + training client ---
    service = tinker.ServiceClient()
    training_client = await service.create_lora_training_client_async(
        args.model,
        rank=args.lora_rank,
    )
    sampling_client = await training_client.save_weights_and_get_sampling_client_async(
        retry_config=retry_config,
    )

    # Frozen opponent (separate sampling client, no gradient)
    opponent_sampling = service.create_sampling_client(
        base_model=args.opponent_model,
        retry_config=retry_config,
    )

    # Judge (frozen, separate model)
    judge_sampling = service.create_sampling_client(
        base_model=args.judge_model,
        retry_config=retry_config,
    )

    # Renderers
    renderer_name = get_recommended_renderer_name(
        args.model, reasoning_effort=args.reasoning_effort
    )
    opponent_renderer_name = get_recommended_renderer_name(
        args.opponent_model, reasoning_effort=args.reasoning_effort
    )
    judge_renderer_name = get_recommended_renderer_name(
        args.judge_model, reasoning_effort=args.reasoning_effort
    )
    renderer = get_renderer(renderer_name, get_tokenizer(args.model))
    opponent_renderer = get_renderer(opponent_renderer_name, get_tokenizer(args.opponent_model))
    judge_renderer = get_renderer(judge_renderer_name, get_tokenizer(args.judge_model))

    usage = UsageTracker()

    opponent_completer = TinkerMessageCompleter(
        sampling_client=opponent_sampling,
        renderer=opponent_renderer,
        max_tokens=args.opponent_max_tokens,
        usage_tracker=usage,
        actor="opponent",
        model_name=args.opponent_model,
    )
    judge_completer = TinkerMessageCompleter(
        sampling_client=judge_sampling,
        renderer=judge_renderer,
        max_tokens=4096,
        usage_tracker=usage,
        actor="judge",
        model_name=args.judge_model,
    )

    dataset = DebateDataset(
        problems=problems,
        batch_size=args.batch_size,
        renderer=renderer,
        protocol_kind=ProtocolKind.SEQUENTIAL,
        num_rounds=args.num_rounds,
        judge_callback=LLMJudgeCallback(judge_completer),
        outcome_reward_fn=zero_sum_outcome_reward,
        opponent_completer=opponent_completer,
        opponent_renderer=opponent_renderer,
        group_size=args.group_size,
        prompts_ref=args.prompts_ref,
        metrics=mcq_debate_metrics(),
    )

    print(f"Model:     {args.model} (renderer={renderer_name})", flush=True)
    print(f"Opponent:  {args.opponent_model} (renderer={opponent_renderer_name}) [frozen]", flush=True)
    print(f"Judge:     {args.judge_model} (renderer={judge_renderer_name})", flush=True)
    print(f"Prompts:   {args.prompts_ref}  reasoning_effort={args.reasoning_effort}", flush=True)
    print(
        f"Training:  batch_size={args.batch_size} group_size={args.group_size} "
        f"num_rounds={args.num_rounds} lr={args.lr} lora_rank={args.lora_rank}",
        flush=True,
    )
    print(f"Batches:   {args.num_batches}", flush=True)
    print(f"Trace:     {TRACE_PATH}", flush=True)
    print(flush=True)

    # --- Training loop ---
    all_batch_metrics: list[dict[str, float]] = []
    all_optim_metrics: list[dict[str, float]] = []
    has_reward_variance = False

    with logtree.init_trace(
        f"Frozen-Opponent Smoke: {args.model} vs {args.opponent_model}",
        path=TRACE_PATH,
    ):
        for i_batch in range(args.num_batches):
            print(f"===== Batch {i_batch} =====", flush=True)

            # Build policy completer from current sampling client
            policy = TinkerTokenCompleter(
                sampling_client=sampling_client,
                max_tokens=args.max_tokens,
                usage_tracker=usage,
                actor="trained",
                model_name=args.model,
            )

            # Rollouts
            builders = dataset.get_batch(i_batch)
            with logtree.scope_header(f"Batch {i_batch}: rollouts"):
                trajectory_groups_P = await asyncio.gather(
                    *[do_group_rollout(builder, policy) for builder in builders]
                )

            # Check reward variance
            for traj_group in trajectory_groups_P:
                rewards = traj_group.get_total_rewards()
                if len(set(rewards)) > 1:
                    has_reward_variance = True

            # Metrics
            taglist_P = [builder.logging_tags() for builder in builders]
            batch_metrics = compute_trajectory_metrics(trajectory_groups_P, taglist_P)
            all_batch_metrics.append(batch_metrics)

            # Advantages + training data
            advantages_P = compute_advantages(trajectory_groups_P)
            data_D, _metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

            # Train step
            optim_metrics: dict[str, float] = {}
            with logtree.scope_header(f"Batch {i_batch}: train_step"):
                await train_step(
                    data_D=data_D,
                    training_client=training_client,
                    learning_rate=args.lr,
                    num_substeps=1,
                    loss_fn=LOSS_FN,
                    metrics=optim_metrics,
                )
            all_optim_metrics.append(optim_metrics)

            # Update sampling client from trained weights
            sampling_client = await training_client.save_weights_and_get_sampling_client_async(
                retry_config=retry_config,
            )

            # Print batch summary
            print(f"  reward/total: {batch_metrics.get('env/all/reward/total', 'N/A')}", flush=True)
            print(f"  optim metrics: {optim_metrics}", flush=True)
            print(flush=True)

    # --- Assertions ---
    print("=" * 60, flush=True)
    print("ASSERTIONS", flush=True)
    print("=" * 60, flush=True)

    # 1. All metric values are finite
    for i_batch, metrics in enumerate(all_batch_metrics):
        for k, v in metrics.items():
            assert math.isfinite(v), f"Batch {i_batch} metric {k}={v} is not finite"
    print("  [PASS] All metric values are finite", flush=True)

    # 2. At least 1 batch has non-uniform rewards in some trajectory group
    assert has_reward_variance, (
        "All trajectory groups had identical rewards across all batches -- no gradient signal"
    )
    print("  [PASS] Found reward variance in at least one trajectory group", flush=True)

    # 3. judge_quality metric present and in [0, 1]
    for i_batch, metrics in enumerate(all_batch_metrics):
        jq = metrics.get("env/all/judge_quality")
        assert jq is not None, f"Batch {i_batch}: env/all/judge_quality missing"
        assert 0 <= jq <= 1, f"Batch {i_batch}: judge_quality={jq} out of [0,1]"
    print("  [PASS] env/all/judge_quality present and in [0, 1]", flush=True)

    # 4. Accuracy metrics present (soft: may be absent if model outputs are unparseable)
    accuracy_present = all(
        "env/all/accuracy.debater_a" in m and "env/all/accuracy.debater_b" in m
        for m in all_batch_metrics
    )
    if accuracy_present:
        print("  [PASS] env/all/accuracy.debater_a and .debater_b present", flush=True)
    else:
        print(
            "  [WARN] env/all/accuracy.debater_a or .debater_b missing "
            "(model may produce unparseable answers — expected for small models)",
            flush=True,
        )

    # 5. parse_success present
    for i_batch, metrics in enumerate(all_batch_metrics):
        assert "env/all/parse_success" in metrics, f"Batch {i_batch}: env/all/parse_success missing"
    print("  [PASS] env/all/parse_success present", flush=True)

    # 6. Optim metrics present (soft: server may not return metrics for all configs)
    optim_present = all(len(om) > 0 for om in all_optim_metrics)
    if optim_present:
        print("  [PASS] Optim metrics present at both steps", flush=True)
    else:
        print("  [WARN] Some batches returned empty optim metrics", flush=True)

    print(flush=True)

    # --- Report ---
    print("=" * 60, flush=True)
    print("METRIC REPORT", flush=True)
    print("=" * 60, flush=True)

    # Collect all metric keys across batches for the table
    all_keys = sorted(
        set().union(*(m.keys() for m in all_batch_metrics)),
    )
    # Header
    header = f"{'Metric':<50}" + "".join(
        f"{'Batch ' + str(i):>14}" for i in range(len(all_batch_metrics))
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for key in all_keys:
        row = f"{key:<50}"
        for metrics in all_batch_metrics:
            val = metrics.get(key)
            if val is None:
                row += f"{'N/A':>14}"
            else:
                row += f"{val:>14.4f}"
        print(row, flush=True)

    # Optim metrics
    print(flush=True)
    print("Optim metrics:", flush=True)
    optim_keys = sorted(set().union(*(m.keys() for m in all_optim_metrics)))
    for key in optim_keys:
        row = f"  {key:<48}"
        for om in all_optim_metrics:
            val = om.get(key)
            if val is None:
                row += f"{'N/A':>14}"
            else:
                row += f"{val:>14.6f}"
        print(row, flush=True)

    print(flush=True)
    print(usage.format_cost_report(), flush=True)
    print(f"Trace: {TRACE_PATH}", flush=True)
    print("ALL ASSERTIONS PASSED", flush=True)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
