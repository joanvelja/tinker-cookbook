import asyncio
import logging
import time
from typing import Any, Sequence

import tinker

from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)
from tinker_cookbook.utils import logtree

logger = logging.getLogger(__name__)

# Max characters for log values in table cells before truncation
LOG_VALUE_MAX_LEN = 100


def _truncate_log_value(value: Any, max_len: int = LOG_VALUE_MAX_LEN) -> tuple[str, bool]:
    """Truncate a log value if it's too long. Returns (display_value, was_truncated)."""
    str_value = str(value)
    if len(str_value) > max_len:
        return str_value[:max_len] + "...", True
    return str_value, False


@logtree.scope_header_decorator
async def do_single_rollout(policy: TokenCompleter, env: Env) -> Trajectory:
    transitions = []
    ob, stop_condition = await env.initial_observation()
    while True:
        ac_with_logprobs = await policy(ob, stop_condition)
        step_result = await env.step(ac_with_logprobs.tokens)
        transition = Transition(
            ob=ob,
            ac=ac_with_logprobs,
            reward=step_result.reward,
            episode_done=step_result.episode_done,
            metrics=step_result.metrics,
            logs=step_result.logs,
        )
        transitions.append(transition)
        ob = step_result.next_observation
        stop_condition = step_result.next_stop_condition
        if step_result.episode_done:
            break
    return Trajectory(transitions=transitions, final_ob=ob)


@logtree.scope_header_decorator
async def do_group_rollout(
    env_group_builder: EnvGroupBuilder, policy: TokenCompleter
) -> TrajectoryGroup:
    envs_G: Sequence[Env] = await env_group_builder.make_envs()
    trajectories_G = await asyncio.gather(*[do_single_rollout(policy, env) for env in envs_G])
    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(trajectories_G, envs_G)
    env_group_builder.on_group_complete(trajectories_G, envs_G, rewards_and_metrics_G)
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)

    # Log trajectory tables with final rewards
    with logtree.scope_header("Trajectory Summary"):
        for i, (traj, final_reward) in enumerate(zip(trajectories_G, rewards_G, strict=True)):
            # Pre-scan to collect all log keys across all transitions (preserving order, deduped)
            all_log_keys = list(dict.fromkeys(key for t in traj.transitions for key in t.logs))

            rows = []
            truncated_values: list[tuple[int, str, str]] = []  # (step, key, full_value)
            step_reward_sum = 0.0
            for t_idx, t in enumerate(traj.transitions):
                step_reward_sum += t.reward
                row: dict[str, Any] = {
                    "step": t_idx,
                    "ob_len": t.ob.length,
                    "ac_len": len(t.ac.tokens),
                    "reward": f"{t.reward:.3f}",
                }
                # Add log fields (user is responsible for avoiding collision with core columns)
                for key in all_log_keys:
                    if key in t.logs:
                        display_val, was_truncated = _truncate_log_value(t.logs[key])
                        row[key] = display_val
                        if was_truncated:
                            truncated_values.append((t_idx, key, str(t.logs[key])))
                    else:
                        row[key] = "-"
                rows.append(row)
            # Add final row with final observation and computed reward
            rows.append(
                {
                    "step": "final",
                    "ob_len": traj.final_ob.length,
                    "ac_len": "-",
                    "reward": f"{final_reward:.3f}",
                    **{key: "-" for key in all_log_keys},
                }
            )
            # Add total reward row
            rows.append(
                {
                    "step": "total",
                    "ob_len": "-",
                    "ac_len": "-",
                    "reward": f"{step_reward_sum + final_reward:.3f}",
                    **{key: "-" for key in all_log_keys},
                }
            )
            logtree.table(rows, caption=f"Trajectory {i}")

            # Show full content for any truncated values in collapsible blocks
            for step_idx, key, full_value in truncated_values:
                logtree.details(
                    full_value,
                    summary=f"Step {step_idx} - {key} (full, {len(full_value)} chars)",
                    pre=True,
                )

    return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G))


@logtree.scope_header_decorator
async def do_batched_group_rollout(
    env_group_builder: "ProblemGroupBuilder",
    sampling_client: tinker.SamplingClient,
    sampling_params: tinker.SamplingParams,
) -> TrajectoryGroup:
    """Batched rollout for single-turn ProblemEnv groups.

    Instead of G separate sample_async(num_samples=1) calls, this makes ONE
    sample_async(num_samples=G) call.  The Tinker backend shares the prompt
    KV-cache across all G decodes, giving 2-4x throughput.

    Requirements:
    - env_group_builder must be a ProblemGroupBuilder (single-turn envs).
    - All envs in the group produce the same prompt (guaranteed by ProblemGroupBuilder).
    """
    from tinker_cookbook.rl.problem_env import ProblemGroupBuilder

    assert isinstance(env_group_builder, ProblemGroupBuilder)
    G = env_group_builder.num_envs

    # 1. Create one env to get the shared prompt
    prompt_env = env_group_builder.env_thunk()
    prompt, stop_condition = await prompt_env.initial_observation()

    # 2. ONE batched sample call
    t0 = time.monotonic()
    sample_response = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=G,
        sampling_params=sampling_params,
    )
    sample_wall_s = time.monotonic() - t0
    logger.debug(
        "Batched sample: G=%d, prompt_len=%d, wall=%.2fs", G, prompt.length, sample_wall_s
    )

    # 3. Create G envs for grading, step each with its sampled tokens
    envs_G: list[Env] = [env_group_builder.env_thunk() for _ in range(G)]
    # Initialize each env so its internal state is ready for step()
    for env in envs_G:
        await env.initial_observation()

    trajectories_G: list[Trajectory] = []
    for env, seq in zip(envs_G, sample_response.sequences, strict=True):
        sampled_tokens = seq.tokens
        sampled_logprobs = seq.logprobs
        assert sampled_logprobs is not None

        ac = TokensWithLogprobs(tokens=sampled_tokens, maybe_logprobs=sampled_logprobs)
        step_result = await env.step(sampled_tokens)

        transition = Transition(
            ob=prompt,
            ac=ac,
            reward=step_result.reward,
            episode_done=step_result.episode_done,
            metrics=step_result.metrics,
            logs=step_result.logs,
        )
        trajectories_G.append(
            Trajectory(
                transitions=[transition],
                final_ob=step_result.next_observation,
            )
        )

    # 4. Compute group rewards (same as do_group_rollout)
    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(trajectories_G, envs_G)
    env_group_builder.on_group_complete(trajectories_G, envs_G, rewards_and_metrics_G)
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)

    # 5. Log trajectory summary (same format as do_group_rollout)
    with logtree.scope_header("Trajectory Summary"):
        for i, (traj, final_reward) in enumerate(zip(trajectories_G, rewards_G, strict=True)):
            all_log_keys = list(dict.fromkeys(key for t in traj.transitions for key in t.logs))

            rows = []
            truncated_values: list[tuple[int, str, str]] = []
            step_reward_sum = 0.0
            for t_idx, t in enumerate(traj.transitions):
                step_reward_sum += t.reward
                row: dict[str, Any] = {
                    "step": t_idx,
                    "ob_len": t.ob.length,
                    "ac_len": len(t.ac.tokens),
                    "reward": f"{t.reward:.3f}",
                }
                for key in all_log_keys:
                    if key in t.logs:
                        display_val, was_truncated = _truncate_log_value(t.logs[key])
                        row[key] = display_val
                        if was_truncated:
                            truncated_values.append((t_idx, key, str(t.logs[key])))
                    else:
                        row[key] = "-"
                rows.append(row)
            rows.append(
                {
                    "step": "final",
                    "ob_len": traj.final_ob.length,
                    "ac_len": "-",
                    "reward": f"{final_reward:.3f}",
                    **{key: "-" for key in all_log_keys},
                }
            )
            rows.append(
                {
                    "step": "total",
                    "ob_len": "-",
                    "ac_len": "-",
                    "reward": f"{step_reward_sum + final_reward:.3f}",
                    **{key: "-" for key in all_log_keys},
                }
            )
            logtree.table(rows, caption=f"Trajectory {i}")

            for step_idx, key, full_value in truncated_values:
                logtree.details(
                    full_value,
                    summary=f"Step {step_idx} - {key} (full, {len(full_value)} chars)",
                    pre=True,
                )

    return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G))
