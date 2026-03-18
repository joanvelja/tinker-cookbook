"""Normalized debate loader — dedup self-play rows, build question/group indices.

Self-play produces 2 rows per debate (one per seat). This module deduplicates
by debate_id, computes question-level and group-level aggregates, and exposes
clean dataclasses for downstream renderers.

Schema v4 episode rows have: debate_id, role, reward, signals, transcript,
winner, task_prompt, target, protocol_kind, group_id, step, etc.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeatRecord:
    role: str
    reward: float
    trajectory_index: int | None
    advantage_subgroup: str | None
    global_row_index: int | None  # position in episodes.jsonl (for episode HTML links)


@dataclass
class Debate:
    debate_id: str
    step: int | None
    split: str | None
    group_id: str | None
    task_prompt: str | None
    target: str | None
    winner: str | None
    verdict_text: str | None
    protocol_kind: str
    prompts_ref: str | None
    answers: dict
    signals: dict
    transcript: list[dict]
    seats: dict[str, SeatRecord]  # keyed by role
    _raw_row: dict  # first row, for replay_debate compat

    @property
    def question_key(self) -> str | None:
        """problem_id > sha256(task_prompt)[:12] > None (skip grouping)."""
        pid = self._raw_row.get("problem_id")
        if pid:
            return pid
        tp = self.task_prompt
        if tp:
            return hashlib.sha256(tp.encode()).hexdigest()[:12]
        return None


@dataclass
class QuestionRecord:
    question_key: str
    task_prompt: str | None
    target: str | None
    protocols: set[str]
    steps: list[int]
    debates: list[Debate]
    n_debates: int
    wrong_winner_rate: float
    draw_rate: float
    interestingness: float


@dataclass
class GroupRecord:
    group_id: str
    raw: dict  # full groups.jsonl record
    debates: list[Debate]
    step: int | None
    task_prompt: str | None
    target: str | None
    removed_before_training: bool
    subgroups: list[dict] | None
    members: list[dict]

    @property
    def is_dead(self) -> bool:
        """All subgroups have zero std → no learning signal."""
        if not self.subgroups:
            return False
        return all(sg.get("std_reward", 1.0) == 0.0 for sg in self.subgroups)

    @property
    def has_miscredit(self) -> bool:
        """Any member has positive advantage but their debate's winner was wrong."""
        debate_map = {d.debate_id: d for d in self.debates}
        for m in self.members:
            adv = m.get("advantage", 0.0)
            if adv <= 0:
                continue
            d = debate_map.get(m.get("debate_id"))
            if d is None:
                continue
            # Check if the winner was the wrong debater
            if d.winner and d.target:
                winner_answer = d.answers.get(f"public_{d.winner}")
                if winner_answer and winner_answer != d.target:
                    return True
        return False


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_rows(path: str | Path) -> list[dict]:
    """Load JSONL, skip corrupt lines."""
    rows = []
    path = Path(path)
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"warning: skipping corrupt line {i} in {path}", file=sys.stderr)
    return rows


# ---------------------------------------------------------------------------
# Dedup: rows → Debate objects
# ---------------------------------------------------------------------------

_DEBATE_LEVEL_FIELDS = ("winner", "task_prompt", "target", "protocol_kind",
                        "verdict_text", "prompts_ref")
# Note: transcript omitted — list comparison is expensive and not worth a warning.


def build_debates(rows: list[dict]) -> list[Debate]:
    """Group rows by debate_id, assert consistency, build Debate objects.

    Rows without debate_id get a synthetic unique ID (no dedup).
    """
    # Store (global_index, row) tuples so we can thread line position through.
    by_id: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for i, row in enumerate(rows):
        did = row.get("debate_id") or f"__synthetic_{i}"
        by_id[did].append((i, row))

    debates = []
    for did, group in by_id.items():
        first_idx, first = group[0]

        # Assert debate-level fields match across rows
        for _, row in group[1:]:
            for fld in _DEBATE_LEVEL_FIELDS:
                v1, v2 = first.get(fld), row.get(fld)
                if v1 != v2:
                    print(
                        f"warning: debate_id={did} field '{fld}' mismatch: "
                        f"{v1!r} vs {v2!r}",
                        file=sys.stderr,
                    )
            # Signals: check keys match (values should be identical for debate-level signals)
            s1, s2 = first.get("signals", {}), row.get("signals", {})
            if set(s1.keys()) != set(s2.keys()):
                print(
                    f"warning: debate_id={did} signals keys mismatch",
                    file=sys.stderr,
                )

        # Build seats
        seats: dict[str, SeatRecord] = {}
        for row_idx, row in group:
            role = row.get("role", "unknown")
            seats[role] = SeatRecord(
                role=role,
                reward=row.get("reward", 0.0),
                trajectory_index=row.get("trajectory_index_in_group"),
                advantage_subgroup=row.get("advantage_subgroup"),
                global_row_index=row_idx,
            )

        debates.append(Debate(
            debate_id=did,
            step=first.get("step"),
            split=first.get("split"),
            group_id=first.get("group_id"),
            task_prompt=first.get("task_prompt"),
            target=first.get("target"),
            winner=first.get("winner"),
            verdict_text=first.get("verdict_text"),
            protocol_kind=first.get("protocol_kind", "unknown"),
            prompts_ref=first.get("prompts_ref"),
            answers=first.get("answers", {}),
            signals=first.get("signals", {}),
            transcript=first.get("transcript", []),
            seats=seats,
            _raw_row=first,
        ))

    return debates


# ---------------------------------------------------------------------------
# Question index
# ---------------------------------------------------------------------------

def build_question_index(debates: list[Debate]) -> dict[str, QuestionRecord]:
    """Group debates by question_key, compute aggregates.

    Debates with question_key=None are skipped.
    """
    by_key: dict[str, list[Debate]] = defaultdict(list)
    for d in debates:
        qk = d.question_key
        if qk is None:
            continue
        by_key[qk].append(d)

    index: dict[str, QuestionRecord] = {}
    for qk, ds in by_key.items():
        n = len(ds)
        # Wrong winner: winner's public answer != target
        n_wrong = 0
        n_draw = 0
        for d in ds:
            if d.winner is None:
                n_draw += 1
            elif d.target:
                winner_answer = d.answers.get(f"public_{d.winner}")
                if winner_answer and winner_answer != d.target:
                    n_wrong += 1

        wrong_rate = n_wrong / n if n else 0.0
        draw_rate = n_draw / n if n else 0.0
        interestingness = wrong_rate * 5 + draw_rate * 1

        protocols = {d.protocol_kind for d in ds}
        steps = sorted({d.step for d in ds if d.step is not None})

        # Use first non-None values for display
        task_prompt = next((d.task_prompt for d in ds if d.task_prompt), None)
        target = next((d.target for d in ds if d.target), None)

        index[qk] = QuestionRecord(
            question_key=qk,
            task_prompt=task_prompt,
            target=target,
            protocols=protocols,
            steps=steps,
            debates=ds,
            n_debates=n,
            wrong_winner_rate=wrong_rate,
            draw_rate=draw_rate,
            interestingness=interestingness,
        )

    return index


# ---------------------------------------------------------------------------
# Group index
# ---------------------------------------------------------------------------

def build_group_index(
    group_rows: list[dict],
    debates: list[Debate],
) -> dict[str, GroupRecord]:
    """Join group records with debates by member debate_id."""
    debate_map = {d.debate_id: d for d in debates}

    index: dict[str, GroupRecord] = {}
    for g in group_rows:
        gid = g.get("group_id")
        if not gid:
            continue

        members = g.get("members", [])
        # Collect debates referenced by this group's members
        seen_dids: set[str] = set()
        group_debates: list[Debate] = []
        for m in members:
            did = m.get("debate_id")
            if did and did not in seen_dids and did in debate_map:
                seen_dids.add(did)
                group_debates.append(debate_map[did])

        index[gid] = GroupRecord(
            group_id=gid,
            raw=g,
            debates=group_debates,
            step=g.get("step"),
            task_prompt=g.get("task_prompt"),
            target=g.get("target"),
            removed_before_training=g.get("removed_before_training", False),
            subgroups=g.get("subgroups"),
            members=members,
        )

    return index
