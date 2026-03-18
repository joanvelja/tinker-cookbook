#!/usr/bin/env python3
"""Probe: check for adjacent same-role messages in build_generation_messages output.

For each of the first 10 episodes, replays every turn where the viewer generates,
calls build_generation_messages, and reports any adjacent same-role pairs.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.replay_debate import (
    advance_state,
    empty_state,
    episode_to_spec,
    turn_to_utterance,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.core.visibility import (
    build_generation_messages,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.types import Role

EPISODES_PATH = Path("logs/protocol-experiment/seq-v2/episodes/episodes.jsonl")
N_EPISODES = 10


def load_episodes(path: Path, n: int) -> list[dict]:
    eps = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            eps.append(json.loads(line))
    return eps


def find_adjacent_same_role(msgs: list[dict]) -> list[tuple[int, str, str, str]]:
    """Return list of (index, role, content_tail_prev, content_head_cur) for adjacent same-role."""
    violations = []
    for i in range(1, len(msgs)):
        if msgs[i]["role"] == msgs[i - 1]["role"]:
            prev_tail = msgs[i - 1]["content"][-120:] if isinstance(msgs[i - 1]["content"], str) else "<non-str>"
            cur_head = msgs[i]["content"][:120] if isinstance(msgs[i]["content"], str) else "<non-str>"
            violations.append((i, msgs[i]["role"], prev_tail, cur_head))
    return violations


def main():
    episodes = load_episodes(EPISODES_PATH, N_EPISODES)
    print(f"Loaded {len(episodes)} episodes from {EPISODES_PATH}\n")

    total_turns = 0
    total_violations = 0

    for ep_idx, ep in enumerate(episodes):
        spec = episode_to_spec(ep)
        state = empty_state(spec)
        schedule = spec.schedule
        transcript = ep["transcript"]
        utterances = [turn_to_utterance(t, schedule) for t in transcript]

        ep_role = Role(ep.get("role", "debater_a"))

        for viewer in [Role.DEBATER_A, Role.DEBATER_B]:
            # Replay from scratch for each viewer
            state = empty_state(spec)
            for i, utt in enumerate(utterances):
                if utt.role == viewer:
                    total_turns += 1
                    msgs, prefill = build_generation_messages(state, viewer)

                    violations = find_adjacent_same_role(msgs)
                    if violations:
                        total_violations += len(violations)
                        print(f"EP {ep_idx} | viewer={viewer.value} | turn {i} "
                              f"(round={utt.round_index}, phase={utt.phase.value}) | "
                              f"{len(violations)} adjacent same-role pair(s):")
                        for vi, role, prev_tail, cur_head in violations:
                            print(f"  msg[{vi-1}] -> msg[{vi}] both '{role}'")
                            print(f"    prev ends:   ...{prev_tail!r}")
                            print(f"    cur starts:  {cur_head!r}...")
                        print()

                # Advance state regardless
                state = advance_state(state, utt)

    print("=" * 80)
    print(f"SUMMARY: {total_turns} generation turns checked, "
          f"{total_violations} adjacent same-role violations found.")


if __name__ == "__main__":
    main()
