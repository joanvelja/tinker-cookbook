#!/usr/bin/env python3
"""Replay a debate episode showing exact model I/O per turn.

Reconstructs the full prompt each debater saw at each turn by replaying
the DebateState and calling build_generation_messages().

Usage:
    # Show debater_a's perspective for episode 0
    uv run python scripts/replay_debate.py logs/.../episodes/episodes.jsonl

    # Pick a specific episode and role
    uv run python scripts/replay_debate.py logs/.../episodes/episodes.jsonl --index 5 --role debater_b

    # JSON output for downstream processing
    uv run python scripts/replay_debate.py logs/.../episodes/episodes.jsonl --json
"""

import argparse
import json
import sys
from dataclasses import replace
from types import MappingProxyType

from tinker_cookbook.recipes.multiplayer_rl.debate.core.schedule import build_schedule
from tinker_cookbook.recipes.multiplayer_rl.debate.core.visibility import build_generation_messages
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateOutcome,
    DebateProblemSpec,
    DebateSpec,
    DebateState,
    Phase,
    ProtocolKind,
    Role,
    ScoringMode,
    ThinkVisibility,
    Utterance,
)


def _load_episode(path: str, index: int) -> dict:
    with open(path) as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"Episode index {index} out of range (file has {i + 1} lines)")


def _episode_to_spec(ep: dict) -> DebateSpec:
    """Reconstruct DebateSpec from episode record."""
    protocol_kind = ProtocolKind(ep["protocol_kind"])

    # Infer num_rounds from transcript
    transcript = ep["transcript"]
    max_round = max((t["round"] for t in transcript), default=0)
    num_rounds = max_round + 1

    schedule = build_schedule(protocol_kind, num_rounds)

    # Reconstruct problem spec
    answers = ep.get("answers", {})
    answer_by_role = {}
    for key, val in answers.items():
        # Keys like "public_debater_a" or "debater_a"
        for role in Role:
            if role.value in key:
                answer_by_role[role] = val
                break

    problem = DebateProblemSpec(
        task_prompt=ep.get("task_prompt", "(task_prompt missing — older episode format)"),
        scoring_mode=ScoringMode.OPEN_ENDED,
        answer_by_role=answer_by_role or None,
        target=ep.get("target"),
    )

    think_vis_raw = ep.get("think_visibility", {})
    think_vis = DebateSpec.decode_think_visibility(think_vis_raw)

    return DebateSpec(
        debate_id=ep.get("debate_id", "unknown"),
        problem=problem,
        schedule=schedule,
        think_visibility=think_vis,
        protocol_kind=protocol_kind,
        prompts_ref=ep.get("prompts_ref", "default"),
    )


def _empty_state(spec: DebateSpec) -> DebateState:
    return DebateState(
        spec=spec,
        slot_index=0,
        rounds_completed=0,
        transcript=(),
        pending_simultaneous=MappingProxyType({}),
        judge_trace=(),
        done=False,
        outcome=None,
    )


def _advance_state(state: DebateState, utt: Utterance) -> DebateState:
    """Append utterance and advance slot/round counters."""
    schedule = state.spec.schedule
    new_transcript = state.transcript + (utt,)

    slot_index = state.slot_index
    rounds_completed = state.rounds_completed

    # Find the slot this utterance belongs to and advance
    if slot_index < len(schedule):
        slot = schedule[slot_index]
        # Check if all actors in this slot have now spoken
        actors_done = {u.role for u in new_transcript if u.slot_id == slot.slot_id}
        if set(slot.actors).issubset(actors_done):
            if slot.boundary_after:
                rounds_completed = slot.round_index + 1
            slot_index += 1

    done = slot_index >= len(schedule)

    return replace(
        state,
        transcript=new_transcript,
        slot_index=slot_index,
        rounds_completed=rounds_completed,
        done=done,
    )


def _turn_to_utterance(turn: dict, schedule: tuple) -> Utterance:
    """Convert episode transcript turn dict to Utterance."""
    role = Role(turn["role"])
    round_idx = turn["round"]
    phase = Phase(turn["phase"])

    # Find matching slot_id from schedule
    slot_id = 0
    for slot in schedule:
        if slot.round_index == round_idx and slot.phase == phase and role in slot.actors:
            slot_id = slot.slot_id
            break

    return Utterance(
        role=role,
        round_index=round_idx,
        phase=phase,
        text=turn["text"],
        token_count=len(turn["text"]) // 4,  # rough estimate
        slot_id=slot_id,
    )


def replay(ep: dict, viewer: Role, *, as_json: bool = False) -> list[dict]:
    """Replay debate from viewer's perspective, returning per-turn I/O."""
    spec = _episode_to_spec(ep)
    state = _empty_state(spec)
    schedule = spec.schedule
    transcript = ep["transcript"]

    results = []
    utterances = [_turn_to_utterance(t, schedule) for t in transcript]

    for i, utt in enumerate(utterances):
        if utt.role == viewer:
            # This is a turn where our viewer generates. Show what they saw.
            msgs, prefill = build_generation_messages(state, viewer)

            turn_record = {
                "turn_index": i,
                "round": utt.round_index,
                "phase": utt.phase.value,
                "role": utt.role.value,
                "input_messages": [{"role": m["role"], "content": m["content"]} for m in msgs],
                "prefill": prefill,
                "output": utt.text,
            }
            results.append(turn_record)

        # Advance state with this utterance
        state = _advance_state(state, utt)

    return results


def _format_message(msg: dict, width: int = 100) -> str:
    role = msg["role"].upper()
    sep = "─" * width
    return f"┌─ {role} {sep[len(role) + 3:]}\n{msg['content']}\n└{sep}"


def print_replay(results: list[dict], ep: dict, viewer: Role) -> None:
    target = ep.get("target", "?")
    winner = ep.get("winner", "tie")
    reward = ep.get("reward", "?")
    answers = ep.get("answers", {})

    print(f"{'═' * 100}")
    print(f"  DEBATE REPLAY — viewing as {viewer.value}")
    print(f"  target={target}  winner={winner}  reward={reward}")
    print(f"  answers: {json.dumps(answers)}")
    print(f"{'═' * 100}\n")

    for turn in results:
        print(f"{'━' * 100}")
        print(
            f"  TURN {turn['turn_index']}  │  "
            f"round={turn['round']}  phase={turn['phase']}  role={turn['role']}"
        )
        print(f"{'━' * 100}")

        print("\n── INPUT (what the model sees) ──\n")
        for msg in turn["input_messages"]:
            print(_format_message(msg))
            print()

        if turn["prefill"]:
            print(f"── PREFILL ──\n{turn['prefill']}\n")

        print("── OUTPUT (what the model produces) ──\n")
        print(turn["output"])
        print()


def main():
    parser = argparse.ArgumentParser(description="Replay debate episode showing exact model I/O.")
    parser.add_argument("episodes_path", help="Path to episodes.jsonl")
    parser.add_argument("--index", "-i", type=int, default=0, help="Episode index (default: 0)")
    parser.add_argument(
        "--role",
        "-r",
        default=None,
        help="Role to view as (debater_a, debater_b). Default: episode's logged role.",
    )
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--list", "-l", action="store_true", help="List episodes with basic info"
    )
    args = parser.parse_args()

    if args.list:
        with open(args.episodes_path) as f:
            for i, line in enumerate(f):
                ep = json.loads(line)
                winner = ep.get("winner", "tie")
                role = ep.get("role", "?")
                reward = ep.get("reward", "?")
                step = ep.get("step", "?")
                split = ep.get("split", "?")
                n_turns = len(ep.get("transcript", []))
                print(
                    f"[{i:4d}] step={step} split={split} role={role} "
                    f"winner={winner} reward={reward} turns={n_turns}"
                )
        return

    ep = _load_episode(args.episodes_path, args.index)

    if args.role:
        viewer = Role(args.role)
    else:
        viewer = Role(ep.get("role", "debater_a"))

    results = replay(ep, viewer)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print_replay(results, ep, viewer)


if __name__ == "__main__":
    main()
