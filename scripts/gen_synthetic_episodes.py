#!/usr/bin/env python3
"""Generate synthetic episodes.jsonl + groups.jsonl for viewer testing.

Produces ~100 episodes covering 10 edge cases for a debate episode viewer
that groups by question and shows GRPO advantage computation.

Usage:
    uv run python scripts/gen_synthetic_episodes.py /tmp/synthetic-viewer-test/
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------

QUESTIONS = {
    "normal_1": "What is the primary mechanism by which prions propagate?",
    "normal_2": "Does the Banach-Tarski paradox require the axiom of choice?",
    "single_obs": "What is the significance of the Langlands program?",
    "dead_group": "Is P equal to NP?",
    "miscredit": "What causes the Mpemba effect?",
    "evolution_0": "What is the role of topoisomerases in DNA replication?",
    "protocol_mismatch": "Can quantum computers break RSA encryption?",
    "missing_pid": "What is dark energy?",
    "seat_asym": "Is the many-worlds interpretation of QM correct?",
    "tie_heavy": "What is consciousness?",
    "removed": "Does free will exist?",
}


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z"


def _debate_id() -> str:
    return uuid.uuid4().hex[:16]


def _transcript(
    n_rounds: int,
    *,
    protocol: str = "sequential",
    a_answer: str = "A",
    b_answer: str = "B",
    brief: bool = True,
) -> list[dict]:
    """Build a short synthetic transcript with n_rounds propose rounds."""
    turns: list[dict] = []
    for r in range(n_rounds):
        if protocol == "simultaneous":
            # Both speak in propose phase
            turns.append({
                "role": "debater_a",
                "phase": "propose",
                "round": r,
                "text": f"Round {r}: I argue the answer is {a_answer}. <answer>{a_answer}</answer>",
                "identity": "debater_a",
            })
            turns.append({
                "role": "debater_b",
                "phase": "propose",
                "round": r,
                "text": f"Round {r}: I argue the answer is {b_answer}. <answer>{b_answer}</answer>",
                "identity": "debater_b",
            })
        else:
            # Sequential: A proposes, B critiques
            turns.append({
                "role": "debater_a",
                "phase": "propose",
                "round": r,
                "text": f"Round {r}: I argue the answer is {a_answer}. <answer>{a_answer}</answer>",
                "identity": "debater_a",
            })
            turns.append({
                "role": "debater_b",
                "phase": "critique",
                "round": r,
                "text": f"Round {r}: My opponent is wrong. The answer is {b_answer}. <answer>{b_answer}</answer>",
                "identity": "debater_b",
            })
    return turns


# ---------------------------------------------------------------------------
# Episode + Group record builders
# ---------------------------------------------------------------------------


@dataclass
class EpisodeBatch:
    """Accumulates episode and group records for one edge case."""
    episodes: list[dict] = field(default_factory=list)
    groups: list[dict] = field(default_factory=list)


def _base_episode(
    *,
    step: int,
    group_id: str,
    group_index: int,
    debate_index: int,
    traj_index: int,
    debate_id: str,
    role: str,
    reward: float,
    target: str | None,
    winner: str | None,
    task_prompt: str,
    transcript: list[dict],
    signals: dict,
    protocol_kind: str = "sequential",
    problem_id: str | None = None,
    verdict_text: str | None = None,
) -> dict:
    record: dict = {
        "schema_version": 4,
        "step": step,
        "split": "train",
        "group_id": group_id,
        "group_index_in_step": group_index,
        "debate_index_in_group": debate_index,
        "trajectory_index_in_group": traj_index,
        "advantage_subgroup": role,
        "timestamp_utc": _ts(),
        "debate_id": debate_id,
        "protocol_kind": protocol_kind,
        "prompts_ref": "default",
        "think_visibility": {"debater_a": "private", "debater_b": "private"},
        "target": target,
        "task_prompt": task_prompt,
        "winner": winner,
        "verdict_text": verdict_text or f"The judge chose {winner}." if winner else "Draw.",
        "role": role,
        "reward": reward,
        "answers": {
            "public_debater_a": "A",
            "public_debater_b": "B",
        },
        "signals": signals,
        "transcript": transcript,
    }
    if problem_id is not None:
        record["problem_id"] = problem_id
    return record


def _base_group(
    *,
    group_id: str,
    step: int,
    group_index: int,
    task_prompt: str,
    target: str | None,
    protocol_kind: str = "sequential",
    members: list[dict],
    subgroups: list[dict] | None = None,
    removed_before_training: bool = False,
    n_debates: int,
) -> dict:
    return {
        "group_id": group_id,
        "step": step,
        "split": "train",
        "group_index_in_step": group_index,
        "task_prompt": task_prompt,
        "target": target,
        "protocol_kind": protocol_kind,
        "advantage_scheme": "grpo",
        "advantage_alpha": 0.0,
        "use_advantage_subgroups": True,
        "removed_before_training": removed_before_training,
        "n_debates": n_debates,
        "n_trajectories": len(members),
        "members": members,
        "subgroups": subgroups,
        "timestamp_utc": _ts(),
    }


def _member(traj_idx: int, debate_idx: int, debate_id: str, role: str, reward: float, advantage: float) -> dict:
    return {
        "trajectory_index": traj_idx,
        "debate_index": debate_idx,
        "debate_id": debate_id,
        "role": role,
        "reward_total": round(reward, 6),
        "advantage": round(advantage, 6),
    }


# ---------------------------------------------------------------------------
# Edge case generators
# ---------------------------------------------------------------------------


def _normal_cases(step_base: int) -> EpisodeBatch:
    """Case 1: Normal — 2 questions, 3 steps each, group_size=4.

    Produces clean question pages with varied outcomes.
    """
    batch = EpisodeBatch()
    questions = [
        ("normal_1", QUESTIONS["normal_1"], "A"),
        ("normal_2", QUESTIONS["normal_2"], "B"),
    ]

    for q_idx, (q_key, q_text, target) in enumerate(questions):
        for step_offset in range(3):
            step = step_base + step_offset
            group_id = f"s{step}:g{q_idx}"
            debate_ids = [_debate_id() for _ in range(4)]
            # Varied outcomes: 2 correct, 1 wrong, 1 draw
            outcomes = [
                ("debater_a", 1.0, -1.0),   # A wins (correct for q1)
                ("debater_a", 1.0, -1.0),   # A wins
                ("debater_b", -1.0, 1.0),   # B wins
                (None, 0.0, 0.0),           # draw
            ]

            members = []
            for d_idx, (winner, r_a, r_b) in enumerate(outcomes):
                did = debate_ids[d_idx]
                tx = _transcript(2, a_answer=target, b_answer="B" if target == "A" else "A")

                signals_a = {
                    "accuracy.debater_a": 1.0 if target == "A" else 0.0,
                    "accuracy.debater_b": 0.0 if target == "A" else 1.0,
                    "win_rate.debater_a": 1.0 if winner == "debater_a" else 0.0,
                    "win_rate.debater_b": 1.0 if winner == "debater_b" else 0.0,
                    "draw_rate": 1.0 if winner is None else 0.0,
                }
                signals_b = dict(signals_a)

                for role_str, reward, signals in [
                    ("debater_a", r_a, signals_a),
                    ("debater_b", r_b, signals_b),
                ]:
                    traj_idx = d_idx * 2 + (0 if role_str == "debater_a" else 1)
                    batch.episodes.append(_base_episode(
                        step=step,
                        group_id=group_id,
                        group_index=q_idx,
                        debate_index=d_idx,
                        traj_index=traj_idx,
                        debate_id=did,
                        role=role_str,
                        reward=reward,
                        target=target,
                        winner=winner,
                        task_prompt=q_text,
                        transcript=tx,
                        signals=signals,
                        problem_id=q_key,
                    ))
                    members.append(_member(traj_idx, d_idx, did, role_str, reward,
                                           0.0))  # placeholder, computed below

            # Compute GRPO advantages within subgroups (A-seat, B-seat).
            a_rewards = [m["reward_total"] for m in members if m["role"] == "debater_a"]
            b_rewards = [m["reward_total"] for m in members if m["role"] == "debater_b"]
            a_mean = sum(a_rewards) / len(a_rewards) if a_rewards else 0.0
            b_mean = sum(b_rewards) / len(b_rewards) if b_rewards else 0.0
            a_std = (sum((r - a_mean) ** 2 for r in a_rewards) / len(a_rewards)) ** 0.5 if a_rewards else 1.0
            b_std = (sum((r - b_mean) ** 2 for r in b_rewards) / len(b_rewards)) ** 0.5 if b_rewards else 1.0

            for m in members:
                if m["role"] == "debater_a":
                    m["advantage"] = round((m["reward_total"] - a_mean) / max(a_std, 1e-8), 6)
                else:
                    m["advantage"] = round((m["reward_total"] - b_mean) / max(b_std, 1e-8), 6)

            a_indices = [m["trajectory_index"] for m in members if m["role"] == "debater_a"]
            b_indices = [m["trajectory_index"] for m in members if m["role"] == "debater_b"]
            subgroups = [
                {"id": 0, "trajectory_indices": a_indices, "mean_reward": round(a_mean, 6), "std_reward": round(a_std, 6)},
                {"id": 1, "trajectory_indices": b_indices, "mean_reward": round(b_mean, 6), "std_reward": round(b_std, 6)},
            ]

            batch.groups.append(_base_group(
                group_id=group_id,
                step=step,
                group_index=q_idx,
                task_prompt=q_text,
                target=target,
                members=members,
                subgroups=subgroups,
                n_debates=4,
            ))

    return batch


def _single_observation(step: int) -> EpisodeBatch:
    """Case 2: Single-observation question — group_size=1 (2 rows = 1 debate).

    Tests n/(n+3) interestingness scaling: with n=1 per subgroup, advantage
    should be heavily discounted.
    """
    batch = EpisodeBatch()
    q = QUESTIONS["single_obs"]
    group_id = f"s{step}:g0"
    did = _debate_id()
    tx = _transcript(1, a_answer="A", b_answer="B")

    for role_str, reward, traj_idx in [("debater_a", 1.0, 0), ("debater_b", -1.0, 1)]:
        batch.episodes.append(_base_episode(
            step=step,
            group_id=group_id,
            group_index=0,
            debate_index=0,
            traj_index=traj_idx,
            debate_id=did,
            role=role_str,
            reward=reward,
            target="A",
            winner="debater_a",
            task_prompt=q,
            transcript=tx,
            signals={
                "accuracy.debater_a": 1.0,
                "accuracy.debater_b": 0.0,
                "win_rate.debater_a": 1.0,
                "win_rate.debater_b": 0.0,
                "draw_rate": 0.0,
            },
            problem_id="single_obs",
        ))

    # n=1 per subgroup → advantage is just sign(reward - mean), but std=0 → raw 0
    members = [
        _member(0, 0, did, "debater_a", 1.0, 0.0),   # sole A → adv=0 (no variance)
        _member(1, 0, did, "debater_b", -1.0, 0.0),  # sole B → adv=0 (no variance)
    ]
    subgroups = [
        {"id": 0, "trajectory_indices": [0], "mean_reward": 1.0, "std_reward": 0.0},
        {"id": 1, "trajectory_indices": [1], "mean_reward": -1.0, "std_reward": 0.0},
    ]
    batch.groups.append(_base_group(
        group_id=group_id, step=step, group_index=0,
        task_prompt=q, target="A",
        members=members, subgroups=subgroups, n_debates=1,
    ))
    return batch


def _dead_group(step: int) -> EpisodeBatch:
    """Case 3: Dead group — all 4 debates have same outcome → zero advantage.

    Tests "dead subgroup" banner: when std=0 for all subgroups, no learning signal.
    """
    batch = EpisodeBatch()
    q = QUESTIONS["dead_group"]
    group_id = f"s{step}:g0"
    debate_ids = [_debate_id() for _ in range(4)]
    tx = _transcript(2, a_answer="A", b_answer="B")

    members = []
    for d_idx in range(4):
        did = debate_ids[d_idx]
        signals = {
            "accuracy.debater_a": 1.0,
            "accuracy.debater_b": 0.0,
            "win_rate.debater_a": 1.0,
            "win_rate.debater_b": 0.0,
            "draw_rate": 0.0,
        }
        for role_str, reward, traj_idx in [
            ("debater_a", 1.0, d_idx * 2),
            ("debater_b", -1.0, d_idx * 2 + 1),
        ]:
            batch.episodes.append(_base_episode(
                step=step, group_id=group_id, group_index=0,
                debate_index=d_idx, traj_index=traj_idx,
                debate_id=did, role=role_str, reward=reward,
                target="A", winner="debater_a", task_prompt=q,
                transcript=tx, signals=signals, problem_id="dead_group",
            ))
            # All same reward per seat → advantage = 0
            members.append(_member(traj_idx, d_idx, did, role_str, reward, 0.0))

    subgroups = [
        {"id": 0, "trajectory_indices": [0, 2, 4, 6], "mean_reward": 1.0, "std_reward": 0.0},
        {"id": 1, "trajectory_indices": [1, 3, 5, 7], "mean_reward": -1.0, "std_reward": 0.0},
    ]
    batch.groups.append(_base_group(
        group_id=group_id, step=step, group_index=0,
        task_prompt=q, target="A",
        members=members, subgroups=subgroups, n_debates=4,
    ))
    return batch


def _miscredit(step: int) -> EpisodeBatch:
    """Case 4: Miscredit — wrong debater wins, gets +1 reward + positive advantage.

    The wrong debater (B defends "B" but target is "A") wins the debate and gets
    positive advantage. Tests miscredit detection in the viewer.
    """
    batch = EpisodeBatch()
    q = QUESTIONS["miscredit"]
    group_id = f"s{step}:g0"
    debate_ids = [_debate_id() for _ in range(4)]

    # 3 debates: B wins (wrong answer wins). 1 debate: A wins (correct).
    # B-seat rewards: [+1, +1, +1, -1] → mean=0.5, the +1s get positive advantage.
    outcomes = [
        ("debater_b", -1.0, 1.0),   # Wrong B wins
        ("debater_b", -1.0, 1.0),   # Wrong B wins
        ("debater_b", -1.0, 1.0),   # Wrong B wins
        ("debater_a", 1.0, -1.0),   # Correct A wins
    ]

    members = []
    for d_idx, (winner, r_a, r_b) in enumerate(outcomes):
        did = debate_ids[d_idx]
        tx = _transcript(2, a_answer="A", b_answer="B")
        # B is wrong (target=A) but wins
        signals = {
            "accuracy.debater_a": 1.0,
            "accuracy.debater_b": 0.0,
            "win_rate.debater_a": 1.0 if winner == "debater_a" else 0.0,
            "win_rate.debater_b": 1.0 if winner == "debater_b" else 0.0,
            "draw_rate": 0.0,
            "wrong_and_wins.debater_a": 0.0,
            "wrong_and_wins.debater_b": 1.0 if winner == "debater_b" else 0.0,
            "correct_and_wins.debater_a": 1.0 if winner == "debater_a" else 0.0,
            "correct_and_wins.debater_b": 0.0,
        }

        for role_str, reward, traj_idx in [
            ("debater_a", r_a, d_idx * 2),
            ("debater_b", r_b, d_idx * 2 + 1),
        ]:
            batch.episodes.append(_base_episode(
                step=step, group_id=group_id, group_index=0,
                debate_index=d_idx, traj_index=traj_idx,
                debate_id=did, role=role_str, reward=reward,
                target="A", winner=winner, task_prompt=q,
                transcript=tx, signals=signals, problem_id="miscredit",
            ))
            members.append(_member(traj_idx, d_idx, did, role_str, reward, 0.0))

    # Compute subgroup advantages
    a_rewards = [m["reward_total"] for m in members if m["role"] == "debater_a"]
    b_rewards = [m["reward_total"] for m in members if m["role"] == "debater_b"]
    a_mean, b_mean = sum(a_rewards) / 4, sum(b_rewards) / 4
    a_std = (sum((r - a_mean) ** 2 for r in a_rewards) / 4) ** 0.5
    b_std = (sum((r - b_mean) ** 2 for r in b_rewards) / 4) ** 0.5

    for m in members:
        if m["role"] == "debater_a":
            m["advantage"] = round((m["reward_total"] - a_mean) / max(a_std, 1e-8), 6)
        else:
            m["advantage"] = round((m["reward_total"] - b_mean) / max(b_std, 1e-8), 6)

    a_indices = [m["trajectory_index"] for m in members if m["role"] == "debater_a"]
    b_indices = [m["trajectory_index"] for m in members if m["role"] == "debater_b"]
    subgroups = [
        {"id": 0, "trajectory_indices": a_indices, "mean_reward": round(a_mean, 6), "std_reward": round(a_std, 6)},
        {"id": 1, "trajectory_indices": b_indices, "mean_reward": round(b_mean, 6), "std_reward": round(b_std, 6)},
    ]
    batch.groups.append(_base_group(
        group_id=group_id, step=step, group_index=0,
        task_prompt=q, target="A",
        members=members, subgroups=subgroups, n_debates=4,
    ))
    return batch


def _cross_step_evolution() -> EpisodeBatch:
    """Case 5: Cross-step evolution — same question at steps 0, 5, 10.

    Accuracy improves 0% → 50% → 100%. Tests step timeline visualization.
    """
    batch = EpisodeBatch()
    q = QUESTIONS["evolution_0"]

    # Step 200: 0% accuracy (wrong debater always wins)
    # Step 205: 50% accuracy (half right)
    # Step 210: 100% accuracy (correct always wins)
    accuracy_by_step = {200: 0.0, 205: 0.5, 210: 1.0}

    for step, acc in accuracy_by_step.items():
        group_id = f"s{step}:g0"
        debate_ids = [_debate_id() for _ in range(4)]
        n_correct = int(acc * 4)

        members = []
        for d_idx in range(4):
            did = debate_ids[d_idx]
            correct_wins = d_idx < n_correct
            winner = "debater_a" if correct_wins else "debater_b"
            r_a = 1.0 if correct_wins else -1.0
            r_b = -r_a
            tx = _transcript(2, a_answer="A", b_answer="B")

            signals = {
                "accuracy.debater_a": 1.0,
                "accuracy.debater_b": 0.0,
                "win_rate.debater_a": 1.0 if winner == "debater_a" else 0.0,
                "win_rate.debater_b": 1.0 if winner == "debater_b" else 0.0,
                "draw_rate": 0.0,
            }

            for role_str, reward, traj_idx in [
                ("debater_a", r_a, d_idx * 2),
                ("debater_b", r_b, d_idx * 2 + 1),
            ]:
                batch.episodes.append(_base_episode(
                    step=step, group_id=group_id, group_index=0,
                    debate_index=d_idx, traj_index=traj_idx,
                    debate_id=did, role=role_str, reward=reward,
                    target="A", winner=winner, task_prompt=q,
                    transcript=tx, signals=signals, problem_id="evolution_0",
                ))
                members.append(_member(traj_idx, d_idx, did, role_str, reward, 0.0))

        # Compute advantages
        a_rewards = [m["reward_total"] for m in members if m["role"] == "debater_a"]
        b_rewards = [m["reward_total"] for m in members if m["role"] == "debater_b"]
        a_mean = sum(a_rewards) / len(a_rewards)
        b_mean = sum(b_rewards) / len(b_rewards)
        a_std = (sum((r - a_mean) ** 2 for r in a_rewards) / len(a_rewards)) ** 0.5
        b_std = (sum((r - b_mean) ** 2 for r in b_rewards) / len(b_rewards)) ** 0.5

        for m in members:
            if m["role"] == "debater_a":
                m["advantage"] = round((m["reward_total"] - a_mean) / max(a_std, 1e-8), 6)
            else:
                m["advantage"] = round((m["reward_total"] - b_mean) / max(b_std, 1e-8), 6)

        a_indices = [m["trajectory_index"] for m in members if m["role"] == "debater_a"]
        b_indices = [m["trajectory_index"] for m in members if m["role"] == "debater_b"]
        subgroups = [
            {"id": 0, "trajectory_indices": a_indices, "mean_reward": round(a_mean, 6), "std_reward": round(a_std, 6)},
            {"id": 1, "trajectory_indices": b_indices, "mean_reward": round(b_mean, 6), "std_reward": round(b_std, 6)},
        ]
        batch.groups.append(_base_group(
            group_id=group_id, step=step, group_index=0,
            task_prompt=q, target="A",
            members=members, subgroups=subgroups, n_debates=4,
        ))

    return batch


def _protocol_mismatch() -> EpisodeBatch:
    """Case 6: Protocol mismatch — same question debated under "sequential" at
    step 0 and "simultaneous" at step 5.

    Tests protocol-aware comparison in the viewer.
    """
    batch = EpisodeBatch()
    q = QUESTIONS["protocol_mismatch"]

    for step, protocol in [(300, "sequential"), (305, "simultaneous")]:
        group_id = f"s{step}:g0"
        debate_ids = [_debate_id() for _ in range(4)]

        members = []
        for d_idx in range(4):
            did = debate_ids[d_idx]
            winner = "debater_a" if d_idx % 2 == 0 else "debater_b"
            r_a = 1.0 if winner == "debater_a" else -1.0
            r_b = -r_a
            tx = _transcript(2, protocol=protocol, a_answer="A", b_answer="B")

            signals = {
                "accuracy.debater_a": 1.0,
                "accuracy.debater_b": 0.0,
                "win_rate.debater_a": 1.0 if winner == "debater_a" else 0.0,
                "win_rate.debater_b": 1.0 if winner == "debater_b" else 0.0,
                "draw_rate": 0.0,
            }

            for role_str, reward, traj_idx in [
                ("debater_a", r_a, d_idx * 2),
                ("debater_b", r_b, d_idx * 2 + 1),
            ]:
                batch.episodes.append(_base_episode(
                    step=step, group_id=group_id, group_index=0,
                    debate_index=d_idx, traj_index=traj_idx,
                    debate_id=did, role=role_str, reward=reward,
                    target="A", winner=winner, task_prompt=q,
                    transcript=tx, signals=signals,
                    protocol_kind=protocol, problem_id="protocol_mismatch",
                ))
                members.append(_member(traj_idx, d_idx, did, role_str, reward, 0.0))

        # 50/50 split → advantages cancel
        for m in members:
            m["advantage"] = 0.0  # balanced, no signal

        a_indices = [m["trajectory_index"] for m in members if m["role"] == "debater_a"]
        b_indices = [m["trajectory_index"] for m in members if m["role"] == "debater_b"]
        subgroups = [
            {"id": 0, "trajectory_indices": a_indices, "mean_reward": 0.0, "std_reward": 1.0},
            {"id": 1, "trajectory_indices": b_indices, "mean_reward": 0.0, "std_reward": 1.0},
        ]
        batch.groups.append(_base_group(
            group_id=group_id, step=step, group_index=0,
            task_prompt=q, target="A",
            members=members, subgroups=subgroups, n_debates=4,
            protocol_kind=protocol,
        ))

    return batch


def _missing_problem_id(step: int) -> EpisodeBatch:
    """Case 7: Missing problem_id — some episodes have it, some don't (legacy).

    Tests fallback grouping by task_prompt when problem_id is absent.
    """
    batch = EpisodeBatch()
    q = QUESTIONS["missing_pid"]
    group_id = f"s{step}:g0"
    debate_ids = [_debate_id() for _ in range(4)]

    members = []
    for d_idx in range(4):
        did = debate_ids[d_idx]
        winner = "debater_a"
        tx = _transcript(1, a_answer="A", b_answer="B")
        signals = {
            "accuracy.debater_a": 1.0,
            "accuracy.debater_b": 0.0,
            "win_rate.debater_a": 1.0,
            "win_rate.debater_b": 0.0,
            "draw_rate": 0.0,
        }

        for role_str, reward, traj_idx in [
            ("debater_a", 1.0, d_idx * 2),
            ("debater_b", -1.0, d_idx * 2 + 1),
        ]:
            # Only first 2 debates have problem_id, last 2 don't (legacy format)
            pid = "missing_pid" if d_idx < 2 else None
            batch.episodes.append(_base_episode(
                step=step, group_id=group_id, group_index=0,
                debate_index=d_idx, traj_index=traj_idx,
                debate_id=did, role=role_str, reward=reward,
                target="A", winner=winner, task_prompt=q,
                transcript=tx, signals=signals,
                problem_id=pid,
            ))
            members.append(_member(traj_idx, d_idx, did, role_str, reward, 0.0))

    # All same → dead group
    subgroups = [
        {"id": 0, "trajectory_indices": [0, 2, 4, 6], "mean_reward": 1.0, "std_reward": 0.0},
        {"id": 1, "trajectory_indices": [1, 3, 5, 7], "mean_reward": -1.0, "std_reward": 0.0},
    ]
    batch.groups.append(_base_group(
        group_id=group_id, step=step, group_index=0,
        task_prompt=q, target="A",
        members=members, subgroups=subgroups, n_debates=4,
    ))
    return batch


def _seat_asymmetry(step: int) -> EpisodeBatch:
    """Case 8: Self-play seat asymmetry — A-seat always wins, B-seat always loses.

    Tests seat_asymmetry signal: the viewer should flag this as a protocol fairness issue.
    """
    batch = EpisodeBatch()
    q = QUESTIONS["seat_asym"]
    group_id = f"s{step}:g0"
    debate_ids = [_debate_id() for _ in range(4)]

    members = []
    for d_idx in range(4):
        did = debate_ids[d_idx]
        # A always wins regardless of correctness
        winner = "debater_a"
        tx = _transcript(2, a_answer="A", b_answer="B")
        signals = {
            "accuracy.debater_a": 0.5,  # not always correct
            "accuracy.debater_b": 0.5,
            "win_rate.debater_a": 1.0,
            "win_rate.debater_b": 0.0,
            "draw_rate": 0.0,
        }

        for role_str, reward, traj_idx in [
            ("debater_a", 1.0, d_idx * 2),
            ("debater_b", -1.0, d_idx * 2 + 1),
        ]:
            batch.episodes.append(_base_episode(
                step=step, group_id=group_id, group_index=0,
                debate_index=d_idx, traj_index=traj_idx,
                debate_id=did, role=role_str, reward=reward,
                target="A", winner=winner, task_prompt=q,
                transcript=tx, signals=signals, problem_id="seat_asym",
            ))
            members.append(_member(traj_idx, d_idx, did, role_str, reward, 0.0))

    # All A seats get +1, all B seats get -1 → dead subgroups (no variance within seat)
    subgroups = [
        {"id": 0, "trajectory_indices": [0, 2, 4, 6], "mean_reward": 1.0, "std_reward": 0.0},
        {"id": 1, "trajectory_indices": [1, 3, 5, 7], "mean_reward": -1.0, "std_reward": 0.0},
    ]
    batch.groups.append(_base_group(
        group_id=group_id, step=step, group_index=0,
        task_prompt=q, target="A",
        members=members, subgroups=subgroups, n_debates=4,
    ))
    return batch


def _removed_group(step: int) -> EpisodeBatch:
    """Case 9: Removed group — group with removed_before_training=true.

    Tests removed banner: these episodes were generated but excluded from the
    training batch (e.g., because all advantages were zero or the group was
    flagged for removal).
    """
    batch = EpisodeBatch()
    q = QUESTIONS["removed"]
    group_id = f"s{step}:g0"
    debate_ids = [_debate_id() for _ in range(4)]

    members = []
    for d_idx in range(4):
        did = debate_ids[d_idx]
        winner = "debater_a" if d_idx < 2 else "debater_b"
        r_a = 1.0 if winner == "debater_a" else -1.0
        r_b = -r_a
        tx = _transcript(1, a_answer="A", b_answer="B")
        signals = {
            "accuracy.debater_a": 1.0,
            "accuracy.debater_b": 0.0,
            "win_rate.debater_a": 1.0 if winner == "debater_a" else 0.0,
            "win_rate.debater_b": 1.0 if winner == "debater_b" else 0.0,
            "draw_rate": 0.0,
        }

        for role_str, reward, traj_idx in [
            ("debater_a", r_a, d_idx * 2),
            ("debater_b", r_b, d_idx * 2 + 1),
        ]:
            batch.episodes.append(_base_episode(
                step=step, group_id=group_id, group_index=0,
                debate_index=d_idx, traj_index=traj_idx,
                debate_id=did, role=role_str, reward=reward,
                target="A", winner=winner, task_prompt=q,
                transcript=tx, signals=signals, problem_id="removed",
            ))
            members.append(_member(traj_idx, d_idx, did, role_str, reward, 0.0))

    # Advantages would be non-zero, but group is removed
    a_rewards = [m["reward_total"] for m in members if m["role"] == "debater_a"]
    b_rewards = [m["reward_total"] for m in members if m["role"] == "debater_b"]
    a_mean = sum(a_rewards) / len(a_rewards)
    b_mean = sum(b_rewards) / len(b_rewards)
    a_std = (sum((r - a_mean) ** 2 for r in a_rewards) / len(a_rewards)) ** 0.5
    b_std = (sum((r - b_mean) ** 2 for r in b_rewards) / len(b_rewards)) ** 0.5

    for m in members:
        if m["role"] == "debater_a":
            m["advantage"] = round((m["reward_total"] - a_mean) / max(a_std, 1e-8), 6)
        else:
            m["advantage"] = round((m["reward_total"] - b_mean) / max(b_std, 1e-8), 6)

    a_indices = [m["trajectory_index"] for m in members if m["role"] == "debater_a"]
    b_indices = [m["trajectory_index"] for m in members if m["role"] == "debater_b"]
    subgroups = [
        {"id": 0, "trajectory_indices": a_indices, "mean_reward": round(a_mean, 6), "std_reward": round(a_std, 6)},
        {"id": 1, "trajectory_indices": b_indices, "mean_reward": round(b_mean, 6), "std_reward": round(b_std, 6)},
    ]
    batch.groups.append(_base_group(
        group_id=group_id, step=step, group_index=0,
        task_prompt=q, target="A",
        members=members, subgroups=subgroups, n_debates=4,
        removed_before_training=True,
    ))
    return batch


def _tie_heavy(step: int) -> EpisodeBatch:
    """Case 10: Tie-heavy question — 90% draws (9 out of 10 debates).

    Tests draw_rate signal highlighting and tie-dominated advantage computation.
    Uses group_size=10 for a meaningful sample.
    """
    batch = EpisodeBatch()
    q = QUESTIONS["tie_heavy"]
    group_id = f"s{step}:g0"
    debate_ids = [_debate_id() for _ in range(10)]

    members = []
    for d_idx in range(10):
        did = debate_ids[d_idx]
        if d_idx == 0:
            # The one decisive debate: A wins
            winner = "debater_a"
            r_a, r_b = 1.0, -1.0
        else:
            # Draw
            winner = None
            r_a, r_b = 0.0, 0.0

        tx = _transcript(2, a_answer="A", b_answer="B")
        signals = {
            "accuracy.debater_a": 1.0,
            "accuracy.debater_b": 0.0,
            "win_rate.debater_a": 1.0 if winner == "debater_a" else 0.0,
            "win_rate.debater_b": 1.0 if winner == "debater_b" else 0.0,
            "draw_rate": 0.0 if winner else 1.0,
        }

        for role_str, reward, traj_idx in [
            ("debater_a", r_a, d_idx * 2),
            ("debater_b", r_b, d_idx * 2 + 1),
        ]:
            batch.episodes.append(_base_episode(
                step=step, group_id=group_id, group_index=0,
                debate_index=d_idx, traj_index=traj_idx,
                debate_id=did, role=role_str, reward=reward,
                target="A", winner=winner, task_prompt=q,
                transcript=tx, signals=signals, problem_id="tie_heavy",
            ))
            members.append(_member(traj_idx, d_idx, did, role_str, reward, 0.0))

    # Compute subgroup advantages: A gets [1, 0, 0, ..., 0], B gets [-1, 0, ..., 0]
    a_rewards = [m["reward_total"] for m in members if m["role"] == "debater_a"]
    b_rewards = [m["reward_total"] for m in members if m["role"] == "debater_b"]
    a_mean = sum(a_rewards) / len(a_rewards)
    b_mean = sum(b_rewards) / len(b_rewards)
    a_std = (sum((r - a_mean) ** 2 for r in a_rewards) / len(a_rewards)) ** 0.5
    b_std = (sum((r - b_mean) ** 2 for r in b_rewards) / len(b_rewards)) ** 0.5

    for m in members:
        if m["role"] == "debater_a":
            m["advantage"] = round((m["reward_total"] - a_mean) / max(a_std, 1e-8), 6)
        else:
            m["advantage"] = round((m["reward_total"] - b_mean) / max(b_std, 1e-8), 6)

    a_indices = [m["trajectory_index"] for m in members if m["role"] == "debater_a"]
    b_indices = [m["trajectory_index"] for m in members if m["role"] == "debater_b"]
    subgroups = [
        {"id": 0, "trajectory_indices": a_indices, "mean_reward": round(a_mean, 6), "std_reward": round(a_std, 6)},
        {"id": 1, "trajectory_indices": b_indices, "mean_reward": round(b_mean, 6), "std_reward": round(b_std, 6)},
    ]
    batch.groups.append(_base_group(
        group_id=group_id, step=step, group_index=0,
        task_prompt=q, target="A",
        members=members, subgroups=subgroups, n_debates=10,
    ))
    return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    all_episodes: list[dict] = []
    all_groups: list[dict] = []

    def _add(b: EpisodeBatch) -> None:
        all_episodes.extend(b.episodes)
        all_groups.extend(b.groups)

    # Case 1: Normal (steps 100-102, 2 questions × 3 steps × 4 debates × 2 seats = 48 eps)
    _add(_normal_cases(step_base=100))

    # Case 2: Single observation (step 20, 1 debate × 2 seats = 2 eps)
    _add(_single_observation(step=20))

    # Case 3: Dead group (step 30, 4 debates × 2 seats = 8 eps)
    _add(_dead_group(step=30))

    # Case 4: Miscredit (step 40, 4 debates × 2 seats = 8 eps)
    _add(_miscredit(step=40))

    # Case 5: Cross-step evolution (steps 200, 205, 210; 4 debates × 2 seats × 3 steps = 24 eps)
    _add(_cross_step_evolution())

    # Case 6: Protocol mismatch (steps 300, 305; 4 debates × 2 seats × 2 protocols = 16 eps)
    _add(_protocol_mismatch())

    # Case 7: Missing problem_id (step 50, 4 debates × 2 seats = 8 eps)
    _add(_missing_problem_id(step=50))

    # Case 8: Seat asymmetry (step 60, 4 debates × 2 seats = 8 eps)
    _add(_seat_asymmetry(step=60))

    # Case 9: Removed group (step 70, 4 debates × 2 seats = 8 eps)
    _add(_removed_group(step=70))

    # Case 10: Tie-heavy (step 80, 10 debates × 2 seats = 20 eps)
    _add(_tie_heavy(step=80))

    # Write
    episodes_path = os.path.join(out_dir, "episodes.jsonl")
    groups_path = os.path.join(out_dir, "groups.jsonl")

    with open(episodes_path, "w") as f:
        for ep in all_episodes:
            f.write(json.dumps(ep) + "\n")

    with open(groups_path, "w") as f:
        for g in all_groups:
            f.write(json.dumps(g) + "\n")

    # Summary
    n_eps = len(all_episodes)
    n_groups = len(all_groups)
    n_debates = sum(g["n_debates"] for g in all_groups)
    steps = sorted({ep["step"] for ep in all_episodes})
    questions = sorted({ep["task_prompt"] for ep in all_episodes})
    pids_present = sum(1 for ep in all_episodes if "problem_id" in ep)
    pids_missing = n_eps - pids_present

    print(f"Generated {n_eps} episodes, {n_groups} groups, {n_debates} debates")
    print(f"  Steps: {steps}")
    print(f"  Questions: {len(questions)}")
    print(f"  Episodes with problem_id: {pids_present}, without: {pids_missing}")
    print(f"  Written to: {episodes_path}")
    print(f"             {groups_path}")

    # Consistency checks
    _validate(all_episodes, all_groups)


def _validate(episodes: list[dict], groups: list[dict]) -> None:
    """Verify internal consistency."""
    errors = []

    # Check debate_id pairing: each debate_id should appear exactly 2 times
    # (once per seat) within each group.
    # Key on (group_id, task_prompt) since group_id alone isn't globally unique.
    from collections import Counter
    for g in groups:
        group_id = g["group_id"]
        task_prompt = g["task_prompt"]
        group_eps = [
            e for e in episodes
            if e["group_id"] == group_id and e["task_prompt"] == task_prompt
        ]

        # Check debate_id counts
        debate_counts = Counter(e["debate_id"] for e in group_eps)
        for did, count in debate_counts.items():
            if count != 2:
                errors.append(f"Group {group_id}: debate {did} has {count} episodes (expected 2)")

        # Check rewards sum to zero per debate (self-play constraint)
        debates = {}
        for e in group_eps:
            debates.setdefault(e["debate_id"], []).append(e["reward"])
        for did, rewards in debates.items():
            if abs(sum(rewards)) > 1e-6:
                errors.append(f"Group {group_id}: debate {did} rewards don't sum to 0: {rewards}")

        # Check member count matches episodes
        if len(group_eps) != g["n_trajectories"]:
            errors.append(
                f"Group {group_id}: {len(group_eps)} episodes but "
                f"n_trajectories={g['n_trajectories']}"
            )

    if errors:
        print(f"\nVALIDATION ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\nAll consistency checks passed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_dir>")
        sys.exit(1)
    generate(sys.argv[1])
