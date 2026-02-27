"""Smoke test: debate rollouts with galaxy_brain prompts.

Runs 2 batches of debates with frozen opponent + LLM judge.
Config: batch_size=1, group_size=2, 2 rounds (propose + critique),
debaters 2048 output tokens, judge 1024 output tokens.
No training -- just rollouts to verify the prompt system end-to-end.

Writes a detailed HTML trace to /tmp/tinker-examples/smoke_galaxy.html
with per-role I/O for each debate.
"""

from __future__ import annotations

import asyncio

import tinker

from tinker_cookbook.completers import TinkerMessageCompleter, TinkerTokenCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.usage import UsageTracker
from tinker_cookbook.utils import logtree

from ..env import DebateDataset, DebateEnv
from ..scoring.judge import LLMJudgeCallback, zero_sum_outcome_reward
from ..prompts import resolve_prompts
from .trace_fmt import DebateTraceCSSInjector, render_rollout_html
from ..types import ProtocolKind


MODEL = "Qwen/Qwen3-4B-Instruct-2507"
RENDERER = "qwen3_instruct"
PROMPTS_REF = "galaxy_brain"
TRACE_PATH = "/tmp/tinker-examples/smoke_galaxy.html"

PROBLEMS: list[tuple[str, str, str]] = [
    (
        "Is mathematics discovered or invented? Provide your strongest argument.",
        "discovered",
        "invented",
    ),
    (
        "Which would be more transformative for humanity: achieving faster-than-light travel, or achieving biological immortality?",
        "faster-than-light travel",
        "biological immortality",
    ),
    (
        "Is consciousness fundamental to the universe, or an emergent property of computation?",
        "fundamental",
        "emergent",
    ),
    (
        "If you could only preserve one: all of Earth's art, or all of Earth's science -- which should survive?",
        "art",
        "science",
    ),
]


async def run():
    prompts = resolve_prompts(PROMPTS_REF)
    print(f"Loaded prompts: {prompts.source_ref}", flush=True)
    print(f"  Content hash: {prompts.content_hash[:16]}...", flush=True)
    print(f"  System roles: {sorted(prompts.system.keys())}", flush=True)
    print(f"  Judge fields: {prompts.get_field_names('judge', 'final')}", flush=True)
    print(flush=True)

    # Separate ServiceClient per actor to avoid holder-level backoff coupling.
    renderer = get_renderer(RENDERER, get_tokenizer(MODEL))
    usage_tracker = UsageTracker()

    # WARNING: Do not pass seed= to SamplingParams with num_samples > 1.
    # Tinker deduplicates seeded samples within a batch, silently collapsing
    # rollout diversity. Seeds are also not deterministic across separate calls.
    # See thinking-machines-lab/tinker-feedback#79.

    opponent_completer = TinkerMessageCompleter(
        sampling_client=tinker.ServiceClient().create_sampling_client(base_model=MODEL),
        renderer=renderer,
        max_tokens=2048,
        usage_tracker=usage_tracker,
        actor="opponent",
        model_name=MODEL,
    )

    judge_completer = TinkerMessageCompleter(
        sampling_client=tinker.ServiceClient().create_sampling_client(base_model=MODEL),
        renderer=renderer,
        max_tokens=1024,
        usage_tracker=usage_tracker,
        actor="judge",
        model_name=MODEL,
    )
    judge_callback = LLMJudgeCallback(judge_completer)

    dataset = DebateDataset(
        problems=PROBLEMS,
        batch_size=1,
        renderer=renderer,
        protocol_kind=ProtocolKind.SEQUENTIAL,
        num_rounds=2,
        judge_callback=judge_callback,
        outcome_reward_fn=zero_sum_outcome_reward,
        opponent_completer=opponent_completer,
        group_size=2,
        randomize_position=True,
        prompts_ref=PROMPTS_REF,
    )

    # Completer for the trained agent's rollouts.
    trained_completer = TinkerTokenCompleter(
        sampling_client=tinker.ServiceClient().create_sampling_client(base_model=MODEL),
        max_tokens=2048,
        usage_tracker=usage_tracker,
        actor="trained",
        model_name=MODEL,
    )

    print("Starting galaxy brain debate smoke test", flush=True)
    print(f"  Model: {MODEL}", flush=True)
    print(f"  Prompts: {PROMPTS_REF}", flush=True)
    print(f"  Problems: {len(PROBLEMS)}", flush=True)
    print("  Batches: 2", flush=True)
    print(f"  Trace: {TRACE_PATH}", flush=True)
    print(flush=True)

    with logtree.init_trace(f"Debate Smoke Test: {PROMPTS_REF} / {MODEL}", path=TRACE_PATH):
        # Inject debate-specific CSS.
        logtree.log_formatter(DebateTraceCSSInjector())

        logtree.log_text(
            f"Model: {MODEL} | Prompts: {PROMPTS_REF} | "
            f"Protocol: SEQUENTIAL | Rounds: 2 | Group size: 2"
        )

        # Collect all builders across batches, run in parallel.
        batched_builders: list[tuple[int, list]] = []
        for batch_idx in range(2):
            batched_builders.append((batch_idx, dataset.get_batch(batch_idx)))

        all_builders = [b for _, builders in batched_builders for b in builders]

        async def _run_problem(builder):
            envs = await builder.make_envs()
            trajectories = await asyncio.gather(
                *[do_single_rollout(trained_completer, env) for env in envs]
            )
            rewards_and_metrics = await builder.compute_group_rewards(trajectories, envs)
            return envs, trajectories, rewards_and_metrics

        with logtree.scope_disable():
            all_results = await asyncio.gather(*[_run_problem(b) for b in all_builders])

        # Log sequentially with batch structure.
        result_iter = iter(all_results)
        for batch_idx, builders in batched_builders:
            with logtree.scope_header(f"Batch {batch_idx}", class_="lt-section db-batch"):
                for i, builder in enumerate(builders):
                    envs, trajectories, rewards_and_metrics = next(result_iter)
                    with logtree.scope_header(
                        f"Group {i}: {builder.task_prompt[:80]}...",
                        class_="lt-section db-group",
                    ):
                        for env, traj, (reward, _) in zip(envs, trajectories, rewards_and_metrics):
                            assert isinstance(env, DebateEnv)
                            logtree.log_html(render_rollout_html(env, reward))

        # Cost report.
        with logtree.scope_header("Cost Report", class_="lt-section db-cost"):
            logtree.details(usage_tracker.format_cost_report(), summary="Tinker Usage", pre=True)

    print(usage_tracker.format_cost_report(), flush=True)
    print(f"Trace written to: {TRACE_PATH}", flush=True)
    print("Smoke test complete!", flush=True)


if __name__ == "__main__":
    asyncio.run(run())
