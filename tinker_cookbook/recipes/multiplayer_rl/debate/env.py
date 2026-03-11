"""Thin Tinker adapter for the debate environment."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import tinker
from tinker_cookbook.renderers import Renderer, format_content_as_string
from tinker_cookbook.rl.types import (
    Action,
    Env,
    Observation,
    StepResult,
)
from tinker_cookbook.completers import MessageCompleter, StopCondition

from .core.reducer import get_current_slot
from .core.runtime import DebateRuntime
from .core.visibility import build_generation_messages
from .types import (
    Role,
    TurnTicket,
)


@dataclass
class DebateEnv(Env):
    """Single-agent view into a shared debate runtime."""

    role: Role
    runtime: DebateRuntime
    renderer: Renderer
    opponent_completer: MessageCompleter | None = field(default=None, repr=False)
    opponent_renderer: Renderer | None = field(default=None, repr=False)
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
        tokenizer = (self.opponent_renderer or self.renderer).tokenizer
        token_count = len(tokenizer.encode(text))
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
