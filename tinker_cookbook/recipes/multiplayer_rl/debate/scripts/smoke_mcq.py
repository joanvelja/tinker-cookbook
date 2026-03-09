"""Smoke test: scientific MCQ debate with GPQA diamond.

Runs 2 problems × group_size=2 with frozen opponent + LLM judge.
Verifies the full scoring pipeline: field extraction, trajectory queries,
metrics computation, reward flattening.

Writes HTML trace to /tmp/tinker-examples/smoke_mcq.html

Usage:
    export $(cat .env | xargs)
    uv run python -m tinker_cookbook.recipes.multiplayer_rl.debate.smoke_mcq
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

from ..env import DebateEnv
from ..dataset import DebateDataset
from ..scoring.judge import LLMJudgeCallback, zero_sum_outcome_reward
from ..scoring.metrics import mcq_debate_metrics
from .trace_fmt import DebateTraceCSSInjector, render_rollout_html
from ..types import DebateGameSpec, ProtocolKind, Role
from ..data.gpqa import assign_seat_answers, load_gpqa_mcq_problems

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
RENDERER = "qwen3_instruct"
PROMPTS_REF = "scientific_mcq"
TRACE_PATH = "/tmp/tinker-examples/smoke_mcq.html"
N_PROBLEMS = 2


async def run():
    problems = assign_seat_answers(load_gpqa_mcq_problems(N_PROBLEMS))
    print(f"Loaded {len(problems)} GPQA problems", flush=True)
    for i, prob in enumerate(problems):
        abr = prob.answer_by_role or {}
        print(
            f"  [{i}] A={abr.get(Role.DEBATER_A, '')} B={abr.get(Role.DEBATER_B, '')} "
            f"target={prob.target}  {prob.task_prompt[:80]}...",
            flush=True,
        )
    print(flush=True)

    # Separate ServiceClient per actor to avoid holder-level backoff coupling.
    renderer = get_renderer(RENDERER, get_tokenizer(MODEL))
    usage = UsageTracker()

    opponent_completer = TinkerMessageCompleter(
        sampling_client=tinker.ServiceClient().create_sampling_client(base_model=MODEL),
        renderer=renderer,
        max_tokens=2048,
        usage_tracker=usage,
        actor="opponent",
        model_name=MODEL,
    )
    judge_completer = TinkerMessageCompleter(
        sampling_client=tinker.ServiceClient().create_sampling_client(base_model=MODEL),
        renderer=renderer,
        max_tokens=1024,
        usage_tracker=usage,
        actor="judge",
        model_name=MODEL,
    )
    trained_completer = TinkerTokenCompleter(
        sampling_client=tinker.ServiceClient().create_sampling_client(base_model=MODEL),
        max_tokens=2048,
        usage_tracker=usage,
        actor="trained",
        model_name=MODEL,
    )

    game = DebateGameSpec(
        protocol_kind=ProtocolKind.SEQUENTIAL,
        num_rounds=2,
        prompts_ref=PROMPTS_REF,
    )
    dataset = DebateDataset(
        problems=problems,
        batch_size=len(problems),
        group_size=2,
        game=game,
        renderer=renderer,
        judge_callback=LLMJudgeCallback(judge_completer),
        outcome_reward_fn=zero_sum_outcome_reward,
        opponent_completer=opponent_completer,
        metrics=mcq_debate_metrics(),
    )

    print(f"Model: {MODEL}", flush=True)
    print(f"Prompts: {PROMPTS_REF}", flush=True)
    print(f"Trace: {TRACE_PATH}", flush=True)
    print(flush=True)

    with logtree.init_trace(f"MCQ Debate Smoke: {PROMPTS_REF} / {MODEL}", path=TRACE_PATH):
        logtree.log_formatter(DebateTraceCSSInjector())
        logtree.log_text(
            f"Model: {MODEL} | Prompts: {PROMPTS_REF} | "
            f"Protocol: SEQUENTIAL | Rounds: 2 | Group size: 2 | "
            f"Problems: {N_PROBLEMS}"
        )

        builders = dataset.get_batch(0)

        # Run all problems in parallel (A1 scheduling fix).
        async def _run_problem(builder):
            envs = await builder.make_envs()
            trajectories = await asyncio.gather(
                *[do_single_rollout(trained_completer, env) for env in envs]
            )
            rewards_and_metrics = await builder.compute_group_rewards(trajectories, envs)
            return envs, trajectories, rewards_and_metrics

        with logtree.scope_disable():
            results = await asyncio.gather(*[_run_problem(b) for b in builders])

        # Log sequentially.
        for i, (builder, (envs, trajectories, rewards_and_metrics)) in enumerate(
            zip(builders, results)
        ):
            with logtree.scope_header(
                f"Problem {i}: {builder.task_prompt[:80]}...",
                class_="lt-section db-group",
            ):
                for env, traj, (reward, metrics) in zip(envs, trajectories, rewards_and_metrics):
                    assert isinstance(env, DebateEnv)
                    logtree.log_html(render_rollout_html(env, reward))

                for env_idx, (env, (reward, metrics)) in enumerate(zip(envs, rewards_and_metrics)):
                    assert isinstance(env, DebateEnv)
                    print(
                        f"  Problem {i} / env {env_idx} ({env.role.value}): reward={reward:.1f}",
                        flush=True,
                    )
                    for k, v in sorted(metrics.items()):
                        print(f"    {k}: {v}", flush=True)
                    print(flush=True)

        with logtree.scope_header("Cost Report", class_="lt-section db-cost"):
            logtree.details(usage.format_cost_report(), summary="Tinker Usage", pre=True)

    print(usage.format_cost_report(), flush=True)
    print(f"Trace: {TRACE_PATH}", flush=True)


if __name__ == "__main__":
    asyncio.run(run())
