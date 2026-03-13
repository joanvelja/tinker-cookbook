"""Dump verbatim I/O for every turn in a simulated debate.

No API calls. Walks through a schedule, inserts fake utterances, and prints
the exact messages each role would see at each turn boundary. Use this to
verify V2 message assembly is correct.

Usage:
    python -m tinker_cookbook.recipes.multiplayer_rl.debate.scripts.dump_io [--prompts galaxy_brain]
"""

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import replace

from ..prompts import resolve_prompts
from ..core.schedule import build_schedule
from ..types import (
    DebateProblemSpec,
    DebateSpec,
    DebateState,
    Phase,
    ProtocolKind,
    Role,
    ScoringMode,
    Utterance,
)
from ..core.visibility import build_generation_messages, get_visible_messages

# Fake debate utterances (recognizable text for inspection).
_FAKE_TURNS: dict[tuple[Role, int, str], str] = {
    (Role.DEBATER_A, 0, "propose"): (
        "<think>I need to argue that math is discovered. Let me think about Platonic realism...</think>\n"
        "Mathematics is discovered, not invented. The Mandelbrot set existed before Mandelbrot — "
        "he merely found it. Pi's digits were fixed before any human computed them. "
        "Mathematical truths are *necessary* truths, not contingent on human convention."
    ),
    (Role.DEBATER_B, 0, "propose"): (
        "<think>I should counter with the formalist view...</think>\n"
        "Mathematics is invented. Axiom systems are human choices — ZFC, intuitionist logic, "
        "category theory — each produces different 'truths'. The natural numbers are a cognitive "
        "artifact shaped by our neural architecture. Alien mathematics might be unrecognizable."
    ),
    (Role.DEBATER_A, 1, "critique"): (
        "<think>Their argument about axiom systems is strong. I'll pivot to unreasonable effectiveness...</think>\n"
        "My opponent conflates the *language* of mathematics with mathematical reality. "
        "Yes, axiom systems are choices — but they all converge on the same structures. "
        "The unreasonable effectiveness of mathematics in physics (Wigner, 1960) demands explanation: "
        "why would an *invented* system predict physical phenomena with 12-digit precision?"
    ),
    (Role.DEBATER_B, 1, "critique"): (
        "<think>The Wigner argument is compelling but has known counters...</think>\n"
        "Wigner's puzzle dissolves under scrutiny. We *select* the mathematics that fits physics — "
        "survivorship bias. Most mathematical structures have zero physical application. "
        "And 'convergence' is circular: we define mathematical structures to be isomorphism-invariant, "
        "then marvel that they're invariant. The map is not the territory."
    ),
}


def _fake_utt(role: Role, round_index: int, phase: Phase, slot_id: int) -> Utterance:
    key = (role, round_index, phase.value)
    text = _FAKE_TURNS.get(key, f"[{role.value} round {round_index} {phase.value}]")
    return Utterance(
        role=role,
        round_index=round_index,
        phase=phase,
        text=text,
        token_count=len(text.split()),
        slot_id=slot_id,
    )


def _fmt_msg(msg: dict, width: int = 100) -> str:
    role = msg.get("role", "???")
    content = msg.get("content", "")
    header = f"  [{role.upper()}]"
    body = textwrap.indent(content, "    ")
    return f"{header}\n{body}"


def dump_debate(
    prompts_ref: str = "default",
    protocol_kind: ProtocolKind = ProtocolKind.SEQUENTIAL,
    num_rounds: int = 2,
    task_prompt: str = "Is mathematics discovered or invented? Provide your strongest argument.",
    answer_a: str = "discovered",
    answer_b: str = "invented",
) -> None:
    prompts = resolve_prompts(prompts_ref)
    schedule = build_schedule(protocol_kind, num_rounds)
    think_visibility = prompts.get_think_visibility()

    problem = DebateProblemSpec.from_seat_answers(task_prompt, answer_a, answer_b, ScoringMode.MCQ)
    spec = DebateSpec(
        debate_id="dump-io-test",
        problem=problem,
        schedule=schedule,
        think_visibility=think_visibility,
        protocol_kind=protocol_kind,
        prompts_ref=prompts_ref,
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

    sep = "=" * 100
    thin_sep = "-" * 100

    print(sep)
    print("DEBATE I/O DUMP")
    print(f"  prompts_ref:    {prompts_ref}")
    print(f"  protocol:       {protocol_kind.value}")
    print(f"  num_rounds:     {num_rounds}")
    print(f"  think_visibility: {spec.encode_think_visibility()}")
    print(f"  schedule:       {len(schedule)} slots")
    print(f"  task_prompt:    {task_prompt[:80]}...")
    print(sep)
    print()

    # Walk through schedule, building messages at each turn.
    transcript: list[Utterance] = []
    rounds_completed = 0

    for slot_idx, slot in enumerate(schedule):
        for actor in slot.actors:
            # Build state at this point.
            cur_state = replace(
                state,
                slot_index=slot_idx,
                rounds_completed=rounds_completed,
                transcript=tuple(transcript),
            )

            print(sep)
            print(
                f"TURN: slot={slot_idx} | round={slot.round_index} | "
                f"phase={slot.phase.value} | actor={actor.value}"
            )
            print(sep)

            # --- Observation (what get_visible_messages returns) ---
            obs_msgs = get_visible_messages(cur_state, actor)
            print(f"\n  OBSERVATION ({len(obs_msgs)} messages):")
            print(thin_sep)
            for m in obs_msgs:
                print(_fmt_msg(m))
            print()

            # --- Full generation prompt (with instructions + prefill) ---
            gen_msgs, prefill = build_generation_messages(cur_state, actor)
            print(f"  GENERATION PROMPT ({len(gen_msgs)} messages, prefill={prefill!r}):")
            print(thin_sep)
            for m in gen_msgs:
                print(_fmt_msg(m))
            print()

            # --- Fake response ---
            utt = _fake_utt(actor, slot.round_index, slot.phase, slot.slot_id)
            print(f"  RESPONSE ({utt.token_count} tokens):")
            print(thin_sep)
            print(textwrap.indent(utt.text, "    "))
            print()

            transcript.append(utt)

        if slot.boundary_after:
            rounds_completed += 1

    # --- Judge final ---
    final_state = replace(
        state,
        slot_index=len(schedule),
        rounds_completed=rounds_completed,
        transcript=tuple(transcript),
        done=True,
    )

    print(sep)
    print("JUDGE FINAL VERDICT")
    print(sep)

    judge_msgs, judge_prefill = build_generation_messages(final_state, Role.JUDGE, trigger="final")
    print(f"\n  GENERATION PROMPT ({len(judge_msgs)} messages, prefill={judge_prefill!r}):")
    print(thin_sep)
    for m in judge_msgs:
        print(_fmt_msg(m))
    print()

    # --- JSON dump for programmatic analysis ---
    print(sep)
    print("JSON DUMP (for analysis)")
    print(sep)

    all_turns = []
    transcript_replay: list[Utterance] = []
    rounds_replay = 0

    for slot_idx, slot in enumerate(schedule):
        for actor in slot.actors:
            cur = replace(
                state,
                slot_index=slot_idx,
                rounds_completed=rounds_replay,
                transcript=tuple(transcript_replay),
            )
            gen_msgs, prefill = build_generation_messages(cur, actor)
            utt = _fake_utt(actor, slot.round_index, slot.phase, slot.slot_id)
            all_turns.append(
                {
                    "slot": slot_idx,
                    "round": slot.round_index,
                    "phase": slot.phase.value,
                    "actor": actor.value,
                    "messages": [dict(m) for m in gen_msgs],
                    "prefill": prefill,
                    "response": utt.text,
                }
            )
            transcript_replay.append(utt)
        if slot.boundary_after:
            rounds_replay += 1

    # Judge
    final = replace(
        state,
        slot_index=len(schedule),
        rounds_completed=rounds_replay,
        transcript=tuple(transcript_replay),
        done=True,
    )
    judge_msgs, judge_prefill = build_generation_messages(final, Role.JUDGE, trigger="final")
    all_turns.append(
        {
            "slot": "final",
            "round": "final",
            "phase": "final",
            "actor": "judge",
            "messages": [dict(m) for m in judge_msgs],
            "prefill": judge_prefill,
            "response": None,
        }
    )

    json_path = "/tmp/tinker-examples/dump_io.json"
    import os

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(all_turns, f, indent=2)
    print(f"\nWritten to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump debate I/O verbatim")
    parser.add_argument(
        "--prompts", default="galaxy_brain", help="Prompts ref (default: galaxy_brain)"
    )
    parser.add_argument("--protocol", default="sequential", choices=[k.value for k in ProtocolKind])
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    dump_debate(
        prompts_ref=args.prompts,
        protocol_kind=ProtocolKind(args.protocol),
        num_rounds=args.rounds,
    )
