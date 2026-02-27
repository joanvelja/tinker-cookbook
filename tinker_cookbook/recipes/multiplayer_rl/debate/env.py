"""Thin Tinker adapter and builders for the debate environment."""

from __future__ import annotations

import asyncio
import json
import os
import random
import threading
import time
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

import chz
import tinker
from tinker_cookbook.renderers import Renderer, format_content_as_string, get_renderer
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from tinker_cookbook.completers import MessageCompleter, StopCondition
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .scoring.metrics import MetricFn, _latest_think_answer, mcq_debate_metrics
from .scoring.trajectory import final_answer
from .plugins import JudgeCallback, OutcomeRewardFn, StepRewardFn
from .prompts import check_ab_symmetry, resolve_prompts
from .core.reducer import fork_state, get_current_slot
from .core.runtime import DebateRuntime
from .core.schedule import build_schedule
from .types import (
    DebateSnapshot,
    DebateSpec,
    DebateState,
    ProtocolKind,
    Role,
    TurnTicket,
)
from .core.visibility import build_generation_messages

_EPISODE_LOG_LOCK = threading.Lock()

IDENTITY_REMAP_BASES = [
    "accuracy",
    "stance_change",
    "concession_correctness",
    "debater_accuracy_delta",
    "win_rate",
    "loss_rate",
    "correct_and_wins",
    "correct_and_loses",
    "wrong_and_wins",
    "wrong_and_loses",
    "parse_success",
    "think_block_rate",
    "think_answer_parse_rate",
    "think_public_answer_match",
    "think_correct_public_wrong",
    "think_wrong_public_correct",
]


def _remap_to_identity(m: Metrics, trained_role: Role) -> Metrics:
    """Remap seat-based metrics (debater_a/b) to identity-based (trained/opponent).

    Returns a NEW dict with all original keys plus id/ prefixed identity keys.
    """
    result = dict(m)  # keep originals

    is_a = trained_role == Role.DEBATER_A
    result["id/trained_role_is_a"] = 1.0 if is_a else 0.0

    for base in IDENTITY_REMAP_BASES:
        a_key = f"{base}.debater_a"
        b_key = f"{base}.debater_b"
        trained_key = f"id/{base}.trained"
        opponent_key = f"id/{base}.opponent"

        if is_a:
            if a_key in m:
                result[trained_key] = m[a_key]
            if b_key in m:
                result[opponent_key] = m[b_key]
        else:
            if b_key in m:
                result[trained_key] = m[b_key]
            if a_key in m:
                result[opponent_key] = m[a_key]

    return result


# (task_prompt, answer_a, answer_b) or (task_prompt, answer_a, answer_b, target)
DebateProblem = tuple[str, str, str] | tuple[str, str, str, str]


@dataclass
class DebateEnv(Env):
    """Single-agent view into a shared debate runtime."""

    role: Role
    runtime: DebateRuntime
    renderer: Renderer
    opponent_completer: MessageCompleter | None = field(default=None, repr=False)
    opponent_role: Role | None = None
    _ticket: TurnTicket | None = field(default=None, repr=False)
    _opponent_task: asyncio.Task | None = field(default=None, repr=False)
    _last_initial_obs_wall_s: float = field(default=0.0, repr=False)

    _opponent_wall_s_accum: float = field(default=0.0, repr=False)

    async def _opponent_submit(self) -> None:
        """Get visible messages for the opponent, call the completer, submit."""
        assert self.opponent_completer is not None and self.opponent_role is not None
        ticket = await self.runtime.wait_for_turn(self.opponent_role)
        if ticket is None:
            return
        messages, _prefill = build_generation_messages(self.runtime.state, self.opponent_role)
        t0 = time.monotonic()
        reply = await self.opponent_completer(list(messages))
        self._opponent_wall_s_accum += time.monotonic() - t0
        text = format_content_as_string(reply["content"], separator="")
        token_count = len(self.renderer.tokenizer.encode(text))
        await self.runtime.submit(ticket, text, token_count)

    async def _drive_opponent(self) -> None:
        """Drive the frozen opponent through sequential turns.

        For simultaneous slots, fire the opponent as a background task and return
        immediately so the trained agent can submit concurrently.
        """
        if self.opponent_completer is None or self.opponent_role is None:
            return
        while not self.runtime.state.done:
            slot = get_current_slot(self.runtime.state)
            if slot is None:
                return
            if self.opponent_role not in slot.actors:
                # Not opponent's turn.
                return
            if len(slot.actors) > 1:
                # Simultaneous slot: fire opponent in background and return so
                # the trained agent can participate in the barrier.
                self._opponent_task = asyncio.ensure_future(self._opponent_submit())
                return
            # Sequential opponent turn: run to completion.
            await self._opponent_submit()

    async def _await_opponent_task(self) -> None:
        """Await any pending background opponent task."""
        if self._opponent_task is not None:
            await self._opponent_task
            self._opponent_task = None

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        t0 = time.monotonic()
        # If opponent goes first, drive them before returning our observation.
        await self._drive_opponent()
        self._ticket = await self.runtime.wait_for_turn(self.role)
        if self._ticket is None:
            self._last_initial_obs_wall_s = time.monotonic() - t0
            return tinker.ModelInput.empty(), self.renderer.get_stop_sequences()
        messages, prefill = build_generation_messages(self.runtime.state, self.role)
        self._last_initial_obs_wall_s = time.monotonic() - t0
        return (
            self.renderer.build_generation_prompt(list(messages), prefill=prefill),
            self.renderer.get_stop_sequences(),
        )

    async def step(self, action: Action) -> StepResult:
        t0 = time.monotonic()
        transcript_len_before = len(self.runtime.state.transcript)

        msg, _ok = self.renderer.parse_response(action)
        text = format_content_as_string(msg["content"], separator="")
        assert self._ticket is not None
        result = await self.runtime.submit(self._ticket, text, len(action))
        # Await any background opponent task from a simultaneous slot.
        await self._await_opponent_task()
        episode_done = result.episode_done
        if not episode_done:
            # Drive opponent turns before waiting for our next turn.
            await self._drive_opponent()
            self._ticket = await self.runtime.wait_for_turn(self.role)
            if self._ticket is not None:
                messages, prefill = build_generation_messages(self.runtime.state, self.role)
                next_ob = self.renderer.build_generation_prompt(list(messages), prefill=prefill)
            else:
                # Episode ended while we were waiting for our next turn.
                episode_done = True
                next_ob = tinker.ModelInput.empty()
        else:
            next_ob = tinker.ModelInput.empty()

        # Enrich logs with opponent output tokens and verdict info.
        logs = dict(result.logs)
        new_utterances = self.runtime.state.transcript[transcript_len_before:]
        opponent_utterances = [u for u in new_utterances if u.role != self.role]
        if opponent_utterances:
            total_opp_tokens = sum(u.token_count for u in opponent_utterances)
            logs["opponent_output_tokens"] = total_opp_tokens
            logs["opponent_text"] = opponent_utterances[-1].text
            if len(opponent_utterances) > 1:
                logs["opponent_turns"] = len(opponent_utterances)

        outcome = self.runtime.state.outcome
        if episode_done and outcome is not None:
            logs["verdict"] = outcome.winner.value if outcome.winner else "tie"
            if outcome.verdict_text:
                logs["verdict_text"] = outcome.verdict_text

        logs["time/step_wall_s"] = time.monotonic() - t0
        if self._opponent_wall_s_accum > 0:
            logs["time/opponent_wall_s"] = self._opponent_wall_s_accum
            self._opponent_wall_s_accum = 0.0

        return StepResult(
            reward=result.reward,
            episode_done=episode_done,
            next_observation=next_ob,
            next_stop_condition=result.stop_condition or self.renderer.get_stop_sequences(),
            metrics=result.metrics,
            logs=logs,
        )


@dataclass
class DebateGroupBuilder(EnvGroupBuilder):
    """Builds a group of DebateEnvs sharing one runtime.

    In normal mode (opponent_completer=None), creates one runtime with one env
    per include_roles entry.

    In frozen-opponent mode (opponent_completer set), creates group_size
    independent runtimes, each with a single trained-agent env. The opponent
    is driven by the completer via _drive_opponent(). Simultaneous slots are
    handled by firing the opponent as a background task (asyncio.ensure_future).
    """

    task_prompt: str
    answer_a: str
    answer_b: str
    renderer: Renderer
    protocol_kind: ProtocolKind
    num_rounds: int
    open_reasoning: bool = False
    include_judge_turns: bool = False
    step_reward_fn: StepRewardFn | None = None
    judge_callback: JudgeCallback | None = None
    outcome_reward_fn: OutcomeRewardFn | None = None
    include_roles: tuple[Role, ...] = (Role.DEBATER_A, Role.DEBATER_B)
    group_size: int = 1
    opponent_completer: MessageCompleter | None = None
    randomize_position: bool = False
    prompts_ref: str = "default"
    target: str | None = None
    metrics: dict[str, MetricFn] | None = field(default=None, repr=False)
    episode_log_dir: str | None = None

    # Set after make_envs
    _runtime: DebateRuntime | None = field(default=None, repr=False)
    _runtimes: list[DebateRuntime] = field(default_factory=list, repr=False)

    def _build_answer_by_role(self) -> dict[Role, str] | None:
        """Build answer_by_role mapping, returning None if both answers are empty."""
        if not self.answer_a and not self.answer_b:
            return None
        return {Role.DEBATER_A: self.answer_a, Role.DEBATER_B: self.answer_b}

    def on_group_complete(
        self,
        trajectories_G: list[Trajectory],
        env_group: Sequence[Env],
        rewards_and_metrics_G: list[tuple[float, Metrics]],
    ) -> None:
        if self.episode_log_dir is None:
            return
        if not self._runtimes:
            return  # Only log in frozen-opponent mode

        os.makedirs(self.episode_log_dir, exist_ok=True)
        records: list[str] = []

        for env, (reward, metrics) in zip(env_group, rewards_and_metrics_G, strict=True):
            assert isinstance(env, DebateEnv)
            state = env.runtime.state
            trained_role = env.role
            opponent_role = Role.DEBATER_B if trained_role == Role.DEBATER_A else Role.DEBATER_A

            record = {
                "schema_version": 1,
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z",
                "debate_id": state.spec.debate_id,
                "protocol_kind": state.spec.protocol_kind.value,
                "prompts_ref": state.spec.prompts_ref,
                "open_reasoning": state.spec.open_reasoning,
                "trained_role": trained_role.value,
                "target": state.spec.target,
                "winner": (
                    state.outcome.winner.value if state.outcome and state.outcome.winner else None
                ),
                "reward_trained": reward,
                "answers": {
                    "public_trained": final_answer(state, role=trained_role),
                    "public_opponent": final_answer(state, role=opponent_role),
                    "think_trained": _latest_think_answer(state, trained_role),
                    "think_opponent": _latest_think_answer(state, opponent_role),
                },
                "signals": {k: v for k, v in metrics.items() if k.startswith("id/")},
                "transcript": [
                    {
                        "role": utt.role.value,
                        "phase": utt.phase.value,
                        "round": utt.round_index,
                        "text": utt.text,
                        "identity": "trained" if utt.role == trained_role else "opponent",
                    }
                    for utt in state.transcript
                ],
                "verdict_text": state.outcome.verdict_text if state.outcome else None,
            }
            records.append(json.dumps(record) + "\n")

        log_path = os.path.join(self.episode_log_dir, "episodes.jsonl")
        with _EPISODE_LOG_LOCK:
            with open(log_path, "a") as f:
                f.writelines(records)

    async def make_envs(self) -> Sequence[Env]:
        # Eager validation: fail fast on bad prompts_ref.
        prompts = resolve_prompts(self.prompts_ref)
        if self.randomize_position:
            for w in check_ab_symmetry(prompts):
                warnings.warn(f"randomize_position=True but A/B asymmetry: {w}")

        schedule = build_schedule(
            self.protocol_kind,
            self.num_rounds,
            include_judge_turns=self.include_judge_turns,
        )
        answer_by_role = self._build_answer_by_role()

        if self.opponent_completer is not None:
            # Frozen-opponent mode: validate schedule actors fit in {trained, opponent}.
            all_schedule_roles = {r for slot in schedule for r in slot.actors}
            debater_roles = {Role.DEBATER_A, Role.DEBATER_B}
            extra = all_schedule_roles - debater_roles
            if extra:
                raise ValueError(
                    f"Frozen-opponent mode only drives debater roles, but schedule "
                    f"includes {extra}. Set include_judge_turns=False or use normal mode."
                )
            envs: list[Env] = []
            self._runtimes = []
            for _ in range(self.group_size):
                if self.randomize_position:
                    trained_role = random.choice([Role.DEBATER_A, Role.DEBATER_B])
                else:
                    trained_role = Role.DEBATER_A
                opponent_role = Role.DEBATER_B if trained_role == Role.DEBATER_A else Role.DEBATER_A
                spec = DebateSpec(
                    debate_id=uuid.uuid4().hex,
                    task_prompt=self.task_prompt,
                    answer_by_role=answer_by_role,
                    schedule=schedule,
                    open_reasoning=self.open_reasoning,
                    protocol_kind=self.protocol_kind,
                    prompts_ref=self.prompts_ref,
                    target=self.target,
                )
                state = DebateState(
                    spec=spec,
                    slot_index=0,
                    rounds_completed=0,
                    transcript=(),
                    pending_simultaneous={},
                    judge_trace=(),
                    done=False,
                    outcome=None,
                )
                runtime = DebateRuntime(
                    state,
                    step_reward_fn=self.step_reward_fn,
                    judge_callback=self.judge_callback,
                )
                self._runtimes.append(runtime)
                envs.append(
                    DebateEnv(
                        role=trained_role,
                        runtime=runtime,
                        renderer=self.renderer,
                        opponent_completer=self.opponent_completer,
                        opponent_role=opponent_role,
                    )
                )
            return envs

        # Normal (multi-agent) mode.
        # Validate all schedule actors have a corresponding env to prevent deadlock.
        required_roles = {r for slot in schedule for r in slot.actors}
        missing = required_roles - set(self.include_roles)
        if missing:
            raise ValueError(
                f"Schedule requires roles {missing} but include_roles={self.include_roles}. "
                f"All schedule actors must have an env, otherwise the runtime deadlocks."
            )
        spec = DebateSpec(
            debate_id=uuid.uuid4().hex,
            task_prompt=self.task_prompt,
            answer_by_role=answer_by_role,
            schedule=schedule,
            open_reasoning=self.open_reasoning,
            protocol_kind=self.protocol_kind,
            prompts_ref=self.prompts_ref,
            target=self.target,
        )
        state = DebateState(
            spec=spec,
            slot_index=0,
            rounds_completed=0,
            transcript=(),
            pending_simultaneous={},
            judge_trace=(),
            done=False,
            outcome=None,
        )
        self._runtime = DebateRuntime(
            state,
            step_reward_fn=self.step_reward_fn,
            judge_callback=self.judge_callback,
        )
        return [
            DebateEnv(role=role, runtime=self._runtime, renderer=self.renderer)
            for role in self.include_roles
        ]

    def _compute_metrics(self, state: DebateState) -> Metrics:
        """Compute flat metrics dict from the current state."""
        metric_fns = self.metrics
        if metric_fns is None:
            metric_fns = mcq_debate_metrics()
        results = {name: fn(state) for name, fn in metric_fns.items()}
        return {name: r.value for name, r in results.items() if r.value is not None}

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        if self.outcome_reward_fn is None:
            # Still compute metrics even without outcome reward.
            if self._runtimes:
                results = []
                for env in env_group:
                    assert isinstance(env, DebateEnv)
                    m = self._compute_metrics(env.runtime.state)
                    m = _remap_to_identity(m, env.role)
                    results.append((0.0, m))
                return results
            if self._runtime is not None:
                m = self._compute_metrics(self._runtime.state)
                return [(0.0, m) for _ in trajectory_group]
            return [(0.0, {}) for _ in trajectory_group]

        if self._runtimes:
            # Frozen-opponent mode: each env has its own runtime.
            results = []
            for env in env_group:
                assert isinstance(env, DebateEnv)
                m = self._compute_metrics(env.runtime.state)
                m = _remap_to_identity(m, env.role)
                outcome = env.runtime.state.outcome
                if outcome is None:
                    results.append((0.0, m))
                else:
                    rewards_by_role = self.outcome_reward_fn(outcome)
                    results.append((rewards_by_role.get(env.role, 0.0), m))
            return results

        # Normal mode: single shared runtime.
        if self._runtime is None:
            return [(0.0, {}) for _ in trajectory_group]
        outcome = self._runtime.state.outcome
        m = self._compute_metrics(self._runtime.state)
        if outcome is None:
            return [(0.0, m) for _ in trajectory_group]
        rewards_by_role = self.outcome_reward_fn(outcome)
        results = []
        for env in env_group:
            assert isinstance(env, DebateEnv)
            reward = rewards_by_role.get(env.role, 0.0)
            results.append((reward, m))
        return results

    def logging_tags(self) -> list[str]:
        return ["debate", self.protocol_kind.value]


@dataclass
class DebateBranchGroupBuilder(EnvGroupBuilder):
    """Builds envs from a DebateSnapshot (for branching/MCTS)."""

    snapshot: DebateSnapshot
    renderer: Renderer
    step_reward_fn: StepRewardFn | None = None
    judge_callback: JudgeCallback | None = None
    outcome_reward_fn: OutcomeRewardFn | None = None
    include_roles: tuple[Role, ...] = (Role.DEBATER_A, Role.DEBATER_B)
    metrics: dict[str, MetricFn] | None = field(default=None, repr=False)

    _runtime: DebateRuntime | None = field(default=None, repr=False)

    async def make_envs(self) -> Sequence[Env]:
        forked = fork_state(self.snapshot.state)
        self._runtime = DebateRuntime(
            forked,
            step_reward_fn=self.step_reward_fn,
            judge_callback=self.judge_callback,
        )
        return [
            DebateEnv(role=role, runtime=self._runtime, renderer=self.renderer)
            for role in self.include_roles
        ]

    def _compute_metrics(self, state: DebateState) -> Metrics:
        metric_fns = self.metrics
        if metric_fns is None:
            metric_fns = mcq_debate_metrics()
        results = {name: fn(state) for name, fn in metric_fns.items()}
        return {name: r.value for name, r in results.items() if r.value is not None}

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        if self._runtime is None:
            return [(0.0, {}) for _ in trajectory_group]
        m = self._compute_metrics(self._runtime.state)
        if self.outcome_reward_fn is None:
            return [(0.0, m) for _ in trajectory_group]
        outcome = self._runtime.state.outcome
        if outcome is None:
            return [(0.0, m) for _ in trajectory_group]
        rewards_by_role = self.outcome_reward_fn(outcome)
        results = []
        for env in env_group:
            assert isinstance(env, DebateEnv)
            reward = rewards_by_role.get(env.role, 0.0)
            results.append((reward, m))
        return results

    def logging_tags(self) -> list[str]:
        return ["debate", "branch", self.snapshot.protocol_kind.value]


class DebateDataset(RLDataset):
    """Dataset of debate problems."""

    def __init__(
        self,
        problems: list[DebateProblem],
        batch_size: int,
        renderer: Renderer,
        protocol_kind: ProtocolKind,
        num_rounds: int,
        open_reasoning: bool = False,
        include_judge_turns: bool = False,
        step_reward_fn: StepRewardFn | None = None,
        judge_callback: JudgeCallback | None = None,
        outcome_reward_fn: OutcomeRewardFn | None = None,
        include_roles: tuple[Role, ...] = (Role.DEBATER_A, Role.DEBATER_B),
        group_size: int = 1,
        opponent_completer: MessageCompleter | None = None,
        randomize_position: bool = False,
        prompts_ref: str = "default",
        metrics: dict[str, MetricFn] | None = None,
        episode_log_dir: str | None = None,
    ) -> None:
        self.problems = problems
        self.batch_size = batch_size
        self.renderer = renderer
        self.protocol_kind = protocol_kind
        self.num_rounds = num_rounds
        self.open_reasoning = open_reasoning
        self.include_judge_turns = include_judge_turns
        self.step_reward_fn = step_reward_fn
        self.judge_callback = judge_callback
        self.outcome_reward_fn = outcome_reward_fn
        self.include_roles = include_roles
        self.group_size = group_size
        self.opponent_completer = opponent_completer
        self.randomize_position = randomize_position
        self.prompts_ref = prompts_ref
        self.metrics = metrics
        self.episode_log_dir = episode_log_dir

    @staticmethod
    def _unpack_problem(problem: DebateProblem) -> tuple[str, str, str, str | None]:
        if len(problem) == 4:
            return problem[0], problem[1], problem[2], problem[3]
        return problem[0], problem[1], problem[2], None

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        if not self.problems:
            return []
        start = (index * self.batch_size) % len(self.problems)
        batch_problems = [
            self.problems[(start + i) % len(self.problems)] for i in range(self.batch_size)
        ]
        return [
            DebateGroupBuilder(
                task_prompt=prompt,
                answer_a=ans_a,
                answer_b=ans_b,
                renderer=self.renderer,
                protocol_kind=self.protocol_kind,
                num_rounds=self.num_rounds,
                open_reasoning=self.open_reasoning,
                include_judge_turns=self.include_judge_turns,
                step_reward_fn=self.step_reward_fn,
                judge_callback=self.judge_callback,
                outcome_reward_fn=self.outcome_reward_fn,
                include_roles=self.include_roles,
                group_size=self.group_size,
                opponent_completer=self.opponent_completer,
                randomize_position=self.randomize_position,
                prompts_ref=self.prompts_ref,
                target=target,
                metrics=self.metrics,
                episode_log_dir=self.episode_log_dir,
            )
            for prompt, ans_a, ans_b, target in (self._unpack_problem(p) for p in batch_problems)
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
    prompts_ref: str = "default"
    # Problems supplied externally (not serialized by chz).
    # Each problem is (task_prompt, answer_a, answer_b) or (task_prompt, answer_a, answer_b, target).
    train_problems: list[DebateProblem] = field(default_factory=list)
    test_problems: list[DebateProblem] = field(default_factory=list)

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        train = DebateDataset(
            problems=self.train_problems,
            batch_size=self.batch_size,
            renderer=renderer,
            protocol_kind=self.protocol_kind,
            num_rounds=self.num_rounds,
            open_reasoning=self.open_reasoning,
            include_judge_turns=self.include_judge_turns,
            prompts_ref=self.prompts_ref,
        )
        test = None
        if self.test_problems:
            test = DebateDataset(
                problems=self.test_problems,
                batch_size=len(self.test_problems),
                renderer=renderer,
                protocol_kind=self.protocol_kind,
                num_rounds=self.num_rounds,
                open_reasoning=self.open_reasoning,
                include_judge_turns=self.include_judge_turns,
                prompts_ref=self.prompts_ref,
            )
        return train, test
