"""Single source of truth for loading + formatting GPQA data.

All GPQA data loading goes through this module. Callers that need
``DebateProblemSpec`` call the ``load_*_problems`` helpers; callers that
need Inspect AI ``Sample`` objects use ``problem_to_sample``.
"""

from __future__ import annotations

import random
from typing import Any

from inspect_ai.dataset import Sample

from ..types import DebateProblemSpec, Role, ScoringMode


# ---------------------------------------------------------------------------
# Row loaders — talk to HuggingFace
# ---------------------------------------------------------------------------


def load_gpqa_mcq_rows(
    subset: str = "gpqa_diamond",
    seed: int = 42,
) -> list[dict]:
    """Load raw GPQA MCQ rows, shuffled deterministically."""
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", subset, split="train")
    rows = [ds[i] for i in range(len(ds))]
    random.Random(seed).shuffle(rows)
    return rows


def load_gpqa_open_ended_rows(
    subset: str = "extended",
    split: str = "train",
    seed: int = 42,
    record_ids: list[str] | None = None,
) -> list[dict]:
    """Load raw GPQA open-ended rows.

    If *record_ids* is given, returns exactly those rows in that order.
    Otherwise returns all rows, shuffled deterministically by *seed*.
    """
    from datasets import load_dataset

    ds = load_dataset("joanvelja/gpqa-open-ended", subset, split=split)
    rows = [ds[i] for i in range(len(ds))]

    if record_ids:
        requested = set(record_ids)
        rows = [r for r in rows if r.get("record_id") in requested]
        found = {r.get("record_id") for r in rows}
        missing = requested - found
        if missing:
            raise ValueError(
                f"Missing record_ids: {', '.join(sorted(str(x) for x in missing))}"
            )
        rows.sort(key=lambda r: record_ids.index(str(r["record_id"])))
    else:
        random.Random(seed).shuffle(rows)

    return rows


# ---------------------------------------------------------------------------
# Row → DebateProblemSpec converters
# ---------------------------------------------------------------------------


def mcq_row_to_problem(row: dict, rng: random.Random) -> DebateProblemSpec:
    """Convert a single GPQA MCQ row to a free-debate DebateProblemSpec."""
    correct = row["Correct Answer"]
    wrong = [row[f"Incorrect Answer {i}"] for i in (1, 2, 3)]

    options = [correct] + wrong
    rng.shuffle(options)
    target_label = chr(ord("A") + options.index(correct))

    question = row["Question"]
    option_lines = "\n".join(
        f"{chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options)
    )
    task_prompt = f"{question}\n\n{option_lines}"

    return DebateProblemSpec.from_seat_answers(
        task_prompt, "", "", ScoringMode.MCQ, target=target_label
    )


def open_ended_row_to_problem(row: dict) -> DebateProblemSpec:
    """Convert a single GPQA open-ended row to a DebateProblemSpec."""
    return DebateProblemSpec.from_seat_answers(
        str(row["question"]),
        "",
        "",
        ScoringMode.OPEN_ENDED,
        target=str(row["answer"]),
        metadata={
            "record_id": str(row.get("record_id", "")),
            "domain": str(row.get("domain", "")),
            "subdomain": str(row.get("subdomain", "")),
            "writer_difficulty": row.get("writer_difficulty"),
            "expert_accuracy": row.get("expert_accuracy"),
            "non_expert_accuracy": row.get("non_expert_accuracy"),
            "conversion_type": row.get("conversion_type"),
            "flag": row.get("flag"),
        },
    )


# ---------------------------------------------------------------------------
# Convenience loaders (rows → problems in one call)
# ---------------------------------------------------------------------------


def load_gpqa_mcq_problems(
    n: int | None = None,
    subset: str = "gpqa_diamond",
    seed: int = 42,
) -> list[DebateProblemSpec]:
    """Load GPQA MCQ problems as free-debate DebateProblemSpecs."""
    rows = load_gpqa_mcq_rows(subset=subset, seed=seed)
    if n is not None:
        rows = rows[:n]
    rng = random.Random(seed)
    return [mcq_row_to_problem(row, rng) for row in rows]


def load_gpqa_open_ended_problems(
    subset: str = "extended",
    split: str = "train",
    seed: int = 42,
    record_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[DebateProblemSpec]:
    """Load GPQA open-ended problems as DebateProblemSpecs."""
    rows = load_gpqa_open_ended_rows(
        subset=subset, split=split, seed=seed, record_ids=record_ids
    )
    if limit is not None and not record_ids:
        rows = rows[:limit]
    return [open_ended_row_to_problem(row) for row in rows]


# ---------------------------------------------------------------------------
# Seat answer assignment
# ---------------------------------------------------------------------------


def assign_seat_answers(
    problems: list[DebateProblemSpec], seed: int = 42
) -> list[DebateProblemSpec]:
    """Assign correct answer to seat A, random wrong to seat B.

    Returns new DebateProblemSpec instances (originals are not mutated).
    """
    rng = random.Random(seed)
    out: list[DebateProblemSpec] = []
    for prob in problems:
        target_label = prob.target
        assert target_label is not None, "assign_seat_answers requires target to be set"
        wrong_label = rng.choice(
            [chr(ord("A") + i) for i in range(4) if chr(ord("A") + i) != target_label]
        )
        out.append(
            DebateProblemSpec.from_seat_answers(
                prob.task_prompt,
                target_label,
                wrong_label,
                ScoringMode.MCQ,
                target=target_label,
            )
        )
    return out


# ---------------------------------------------------------------------------
# DebateProblemSpec ↔ Sample conversion
# ---------------------------------------------------------------------------


def problem_to_sample(
    problem: DebateProblemSpec,
    *,
    source: str = "gpqa",
) -> Sample:
    """Convert a DebateProblemSpec to an Inspect AI Sample."""
    abr = problem.answer_by_role or {}
    metadata: dict[str, Any] = {
        "answer_a": abr.get(Role.DEBATER_A, ""),
        "answer_b": abr.get(Role.DEBATER_B, ""),
        "source": source,
    }
    if problem.metadata:
        metadata.update(problem.metadata)

    return Sample(
        input=problem.task_prompt,
        target=problem.target or "",
        metadata=metadata,
    )
