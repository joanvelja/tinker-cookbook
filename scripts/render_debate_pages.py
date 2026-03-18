#!/usr/bin/env python3
"""Render question-level and group-level debate pages.

Reads episodes.jsonl (+ optional groups.jsonl) from a log directory,
deduplicates self-play rows, and generates:
  - Per-question pages (questions/<hash>.html)
  - Per-group pages (groups/<group_id>.html)
  - Top-level index (index.html) linking everything

Usage:
    uv run python scripts/render_debate_pages.py logs/.../episodes/
    uv run python scripts/render_debate_pages.py logs/.../episodes/ -o /tmp/pages
"""

import argparse
import sys
from pathlib import Path

from scripts.debate_loader import (
    Debate,
    GroupRecord,
    QuestionRecord,
    build_debates,
    build_group_index,
    build_question_index,
    load_rows,
)
from scripts.debate_style import _CSS, _JS, _esc, _render_signals_html


# ---------------------------------------------------------------------------
# Group pages
# ---------------------------------------------------------------------------


def _winner_cls(winner: str | None) -> str:
    return {"debater_a": "winner--a", "debater_b": "winner--b"}.get(
        str(winner), "winner--tie"
    )


def _render_failure_banners(group: GroupRecord) -> str:
    """Render conditional failure banners for dead/miscredit/removed groups."""
    parts = []
    if group.is_dead:
        parts.append(
            '<div class="banner-dead">DEAD — all subgroups have zero reward variance; '
            "no learning signal in this group</div>"
        )
    if group.has_miscredit:
        parts.append(
            '<div class="banner-miscredit">MISCREDIT — positive advantage assigned '
            "to a trajectory whose debate winner gave the wrong answer</div>"
        )
    if group.removed_before_training:
        parts.append(
            '<div class="banner-removed">REMOVED — this group was excluded from '
            "training (removed_before_training=true)</div>"
        )
    return "\n".join(parts)


def _render_subgroup_lanes(group: GroupRecord) -> str:
    """Render subgroup lanes with reward/advantage bars."""
    if not group.subgroups:
        return '<div class="dim">no subgroup data</div>'

    # Build member lookup by trajectory_index
    members_by_traj = {m["trajectory_index"]: m for m in group.members}
    debate_map = {d.debate_id: d for d in group.debates}

    parts = []
    for sg in group.subgroups:
        sg_id = sg.get("id", "?")
        mean_r = sg.get("mean_reward", 0.0)
        std_r = sg.get("std_reward", 0.0)
        traj_indices = sg.get("trajectory_indices", [])

        parts.append(f'<div class="g-lane">')
        parts.append(
            f'<div class="g-lane-hdr">subgroup {sg_id} '
            f'— mean: {mean_r:+.3f}, std: {std_r:.3f}, '
            f'n={len(traj_indices)}</div>'
        )

        # Render each member in this subgroup
        for ti in traj_indices:
            m = members_by_traj.get(ti)
            if not m:
                continue
            reward = m.get("reward_total", 0.0)
            advantage = m.get("advantage", 0.0)
            did = m.get("debate_id", "?")
            role = m.get("role", "?")

            # Reward bar (0 to 1 scale, but can be -1 to 1)
            reward_pct = max(0, min(100, (reward + 1) * 50))
            # Advantage bar centered at zero
            adv_abs = min(abs(advantage), 3.0)  # clamp for display
            adv_pct = adv_abs / 3.0 * 50  # max 50% width
            adv_cls = "bar-fill--pos" if advantage >= 0 else "bar-fill--neg"

            debate = debate_map.get(did)
            winner_label = debate.winner if debate else "?"

            parts.append(f'<div class="bar-reward">')
            parts.append(f'<span class="bar-label">{_esc(did[:8])} {_esc(role[:1].upper())}</span>')
            parts.append(f'<div class="bar-fill" style="width:{reward_pct:.0f}%"></div>')
            parts.append(f'<span style="color:var(--text-2);font-size:0.68rem">{reward:+.2f}</span>')
            parts.append("</div>")

            adv_color = "var(--correct)" if advantage >= 0 else "var(--incorrect)"
            adv_margin = "50%" if advantage >= 0 else f"{50 - adv_pct:.0f}%"
            parts.append(f'<div class="bar-advantage">')
            parts.append(f'<span class="bar-label"></span>')
            parts.append(
                f'<div style="display:flex;width:100%;position:relative;height:14px">'
                f'<div style="position:absolute;left:50%;width:1px;height:100%;background:var(--border)"></div>'
                f'<div class="{adv_cls}" style="margin-left:{adv_margin};width:{adv_pct:.0f}%;'
                f'height:14px;border-radius:2px;min-width:2px;background:{adv_color}"></div>'
                f"</div>"
            )
            parts.append(f'<span style="color:var(--text-2);font-size:0.68rem">{advantage:+.3f}</span>')
            parts.append("</div>")

        parts.append("</div>")

    return "\n".join(parts)


def _render_group_debate_cards(group: GroupRecord, html_base: str) -> str:
    """Render debate cards for a group page."""
    parts = []
    for d in group.debates:
        winner = d.winner or "tie"
        w_cls = _winner_cls(d.winner)
        answer_a = d.answers.get("public_debater_a", "?")
        answer_b = d.answers.get("public_debater_b", "?")
        target = d.target or "?"

        a_correct = "correct" if str(answer_a) == str(target) else "incorrect"
        b_correct = "correct" if str(answer_b) == str(target) else "incorrect"

        # Link to episode page (use global JSONL row index, not per-group trajectory index)
        ep_link = ""
        for seat in d.seats.values():
            if seat.global_row_index is not None:
                ep_link = f'{html_base}episode_{seat.global_row_index:04d}.html'
                break

        parts.append('<div style="display:flex;gap:0.5rem;align-items:center;'
                     'padding:0.4rem 0;border-bottom:1px solid var(--border-sub)">')
        parts.append(f'<span class="meta-pill {w_cls}" style="min-width:6rem;text-align:center">'
                     f'{_esc(str(winner))}</span>')
        parts.append(f'<span style="color:var(--role-a);font-size:0.75rem">A: {_esc(str(answer_a)[:30])}'
                     f' <span style="color:var(--{a_correct})">[{a_correct[0]}]</span></span>')
        parts.append(f'<span style="color:var(--role-b);font-size:0.75rem">B: {_esc(str(answer_b)[:30])}'
                     f' <span style="color:var(--{b_correct})">[{b_correct[0]}]</span></span>')
        if ep_link:
            parts.append(f'<a href="{ep_link}" style="color:var(--role-judge);font-size:0.68rem;'
                         f'margin-left:auto">view</a>')
        parts.append("</div>")

    return "\n".join(parts)


def render_group_page(group: GroupRecord, html_base: str = "../episodes/") -> str:
    """Generate self-contained HTML for one group."""
    step = group.step or "?"
    target = group.target or "?"
    task_prompt = group.task_prompt or ""

    # Advantage config from raw record
    scheme = group.raw.get("advantage_scheme", "?")
    alpha = group.raw.get("advantage_alpha", "?")
    use_subgroups = group.raw.get("use_advantage_subgroups", False)

    banners = _render_failure_banners(group)
    lanes = _render_subgroup_lanes(group)
    cards = _render_group_debate_cards(group, html_base)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Group {_esc(group.group_id)} — step {step}</title>
{_CSS}
</head>
<body>

<header class="ep-header">
    <div class="ep-id">group <span class="ep-id-hash">{_esc(group.group_id)}</span></div>
    <div class="ep-question">{_esc(task_prompt) if task_prompt else '<span class="dim">no task_prompt</span>'}</div>
    <div class="ep-meta">
        <span class="meta-pill">step: <b>{step}</b></span>
        <span class="meta-pill">target: <b>{_esc(str(target))}</b></span>
        <span class="meta-pill">{len(group.debates)} debates</span>
        <span class="meta-pill dim">{_esc(scheme)} &alpha;={alpha} subgroups={'yes' if use_subgroups else 'no'}</span>
    </div>
</header>

{banners}

<main style="max-width:960px;padding:1rem 2rem;">
    <h3 style="font-family:var(--font-ui);font-size:0.78rem;color:var(--text-2);margin-bottom:0.5rem;
        letter-spacing:0.04em;text-transform:uppercase">Subgroup Lanes</h3>
    {lanes}

    <h3 style="font-family:var(--font-ui);font-size:0.78rem;color:var(--text-2);margin:1.5rem 0 0.5rem;
        letter-spacing:0.04em;text-transform:uppercase">Debates</h3>
    {cards}
</main>

{_JS}
</body>
</html>"""


def render_groups_section(
    group_index: dict[str, GroupRecord],
) -> str:
    """Render groups table for the top-level index page."""
    if not group_index:
        return ""

    groups = sorted(group_index.values(), key=lambda g: (g.step or 0), reverse=True)
    rows = []
    for g in groups:
        badges = []
        if g.is_dead:
            badges.append('<span class="interestingness-badge interestingness-badge--high">DEAD</span>')
        if g.has_miscredit:
            badges.append('<span class="interestingness-badge interestingness-badge--mid">MISCREDIT</span>')
        if g.removed_before_training:
            badges.append('<span class="interestingness-badge interestingness-badge--low">REMOVED</span>')
        badge_html = " ".join(badges) if badges else ""

        prompt_preview = _esc((g.task_prompt or "")[:50])
        rows.append(
            f"<tr>"
            f'<td><a href="groups/{_esc(g.group_id)}.html">{_esc(g.group_id)}</a></td>'
            f"<td>{g.step or '?'}</td>"
            f"<td>{prompt_preview}</td>"
            f"<td>{len(g.debates)}</td>"
            f"<td>{badge_html}</td>"
            f"</tr>"
        )

    return f"""
    <h2 style="font-family:var(--font-ui);font-size:0.82rem;color:var(--text-2);margin:2rem 0 0.5rem;
        letter-spacing:0.04em;text-transform:uppercase">Groups ({len(groups)})</h2>
    <table class="q-table">
    <thead><tr>
        <th>group</th><th>step</th><th>question</th><th>debates</th><th>flags</th>
    </tr></thead>
    <tbody>
    {"".join(rows)}
    </tbody>
    </table>
    """


# ---------------------------------------------------------------------------
# Question pages
# ---------------------------------------------------------------------------


def _interestingness_badge(score: float, n: int) -> str:
    """Render interestingness badge with tier coloring."""
    if score >= 2.0:
        cls = "interestingness-badge--high"
    elif score >= 0.5:
        cls = "interestingness-badge--mid"
    else:
        cls = "interestingness-badge--low"
    return f'<span class="interestingness-badge {cls}">{score:.1f} (n={n})</span>'


def _step_timeline(question: QuestionRecord) -> str:
    """Render step timeline — dots colored by mean accuracy at each step."""
    if not question.steps:
        return '<span class="dim">no step data</span>'

    # Group debates by step
    by_step: dict[int, list[Debate]] = {}
    for d in question.debates:
        if d.step is not None:
            by_step.setdefault(d.step, []).append(d)

    parts = ['<div class="step-dots">']
    for step in question.steps:
        ds = by_step.get(step, [])
        n_correct = 0
        n_wrong = 0
        n_draw = 0
        for d in ds:
            if d.winner is None:
                n_draw += 1
            elif d.target:
                winner_answer = d.answers.get(f"public_{d.winner}")
                if winner_answer and winner_answer == d.target:
                    n_correct += 1
                else:
                    n_wrong += 1
            else:
                n_correct += 1  # no target → can't judge, count as neutral

        total = n_correct + n_wrong + n_draw
        if total == 0:
            continue

        # Color: majority outcome
        if n_correct >= n_wrong and n_correct >= n_draw:
            cls = "dot--correct"
        elif n_wrong >= n_correct and n_wrong >= n_draw:
            cls = "dot--incorrect"
        else:
            cls = "dot--draw"

        acc = n_correct / total if total else 0
        title = f"step {step}: {n_correct}/{total} correct ({acc:.0%})"
        parts.append(f'<span class="dot {cls}" title="{_esc(title)}"></span>')

    parts.append("</div>")
    return "\n".join(parts)


def _debate_table(question: QuestionRecord, html_base: str) -> str:
    """Render debate table — 1 row per debate (NOT per seat)."""
    # Sort by step then debate_id for stable ordering
    debates = sorted(question.debates, key=lambda d: (d.step or 0, d.debate_id))

    rows = []
    for d in debates:
        step = d.step or "?"
        winner = d.winner or "draw"
        w_cls = _winner_cls(d.winner)

        answer_a = d.answers.get("public_debater_a", "?")
        answer_b = d.answers.get("public_debater_b", "?")
        target = d.target or "?"

        a_correct = str(answer_a) == str(target) if d.target else None
        b_correct = str(answer_b) == str(target) if d.target else None

        a_cls = "correct" if a_correct else ("incorrect" if a_correct is not None else "text-3")
        b_cls = "correct" if b_correct else ("incorrect" if b_correct is not None else "text-3")
        a_mark = "c" if a_correct else ("w" if a_correct is not None else "?")
        b_mark = "c" if b_correct else ("w" if b_correct is not None else "?")

        # Judge quality signal if available
        jq = d.signals.get("judge_quality", d.signals.get("judge_quality.mean", ""))
        jq_str = f"{jq:.2f}" if isinstance(jq, (int, float)) else str(jq) if jq else ""

        # Link to episode page (first seat's global row index)
        ep_link = ""
        for seat in d.seats.values():
            if seat.global_row_index is not None:
                ep_link = f'{html_base}episode_{seat.global_row_index:04d}.html'
                break

        link_cell = f'<a href="{ep_link}">view</a>' if ep_link else ""

        rows.append(
            f"<tr>"
            f"<td>{step}</td>"
            f'<td><span class="meta-pill {w_cls}" style="font-size:0.68rem">{_esc(str(winner))}</span></td>'
            f'<td style="color:var(--role-a)">{_esc(str(answer_a)[:30])} '
            f'<span style="color:var(--{a_cls})">[{a_mark}]</span></td>'
            f'<td style="color:var(--role-b)">{_esc(str(answer_b)[:30])} '
            f'<span style="color:var(--{b_cls})">[{b_mark}]</span></td>'
            f"<td>{jq_str}</td>"
            f"<td>{link_cell}</td>"
            f"</tr>"
        )

    return f"""<table class="q-table">
    <thead><tr>
        <th>step</th><th>winner</th><th>A answer</th><th>B answer</th>
        <th>judge Q</th><th>episode</th>
    </tr></thead>
    <tbody>{"".join(rows)}</tbody>
    </table>"""


def _avg_signals_panel(question: QuestionRecord) -> str:
    """Render collapsed signals panel with debate-grain averages."""
    if not question.debates:
        return ""

    # Collect all signal keys and average numeric values
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for d in question.debates:
        for k, v in d.signals.items():
            if isinstance(v, (int, float)):
                totals[k] = totals.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1

    if not totals:
        return ""

    avg_signals = {k: totals[k] / counts[k] for k in sorted(totals)}
    return _render_signals_html({"signals": avg_signals})


def render_question_page(question: QuestionRecord, html_base: str = "../episodes/") -> str:
    """Generate self-contained HTML for one question."""
    prompt = question.task_prompt or ""
    target = question.target or "?"
    protocols_str = ", ".join(sorted(question.protocols))
    badge = _interestingness_badge(question.interestingness, question.n_debates)
    timeline = _step_timeline(question)
    table = _debate_table(question, html_base)
    signals = _avg_signals_panel(question)
    step_range = f"{question.steps[0]}–{question.steps[-1]}" if question.steps else "?"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Question {_esc(question.question_key)} — {_esc(prompt[:60])}</title>
{_CSS}
</head>
<body>

<header class="ep-header">
    <div class="ep-id">question <span class="ep-id-hash">{_esc(question.question_key)}</span></div>
    <div class="ep-question">{_esc(prompt) if prompt else '<span class="dim">no task_prompt</span>'}</div>
    <div class="ep-meta">
        <span class="meta-pill">target: <b>{_esc(str(target))}</b></span>
        <span class="meta-pill">{question.n_debates} debates</span>
        <span class="meta-pill">steps {step_range}</span>
        <span class="meta-pill dim">{_esc(protocols_str)}</span>
        {badge}
    </div>
</header>

<main style="max-width:960px;padding:1rem 2rem;">
    <h3 style="font-family:var(--font-ui);font-size:0.78rem;color:var(--text-2);margin-bottom:0.5rem;
        letter-spacing:0.04em;text-transform:uppercase">Step Timeline</h3>
    <div style="margin-bottom:1rem;display:flex;align-items:center;gap:0.75rem">
        {timeline}
        <span class="dim" style="font-size:0.68rem">
            wrong: {question.wrong_winner_rate:.0%} &middot; draw: {question.draw_rate:.0%}
        </span>
    </div>

    <h3 style="font-family:var(--font-ui);font-size:0.78rem;color:var(--text-2);margin:1rem 0 0.5rem;
        letter-spacing:0.04em;text-transform:uppercase">Debates</h3>
    {table}

    {signals}
</main>

{_JS}
</body>
</html>"""


def render_questions_section(
    question_index: dict[str, QuestionRecord],
) -> str:
    """Render questions table for the top-level index page."""
    if not question_index:
        return ""
    questions = sorted(question_index.values(), key=lambda q: q.interestingness, reverse=True)
    rows = []
    for q in questions:
        prompt_preview = _esc((q.task_prompt or "")[:50])
        badge = _interestingness_badge(q.interestingness, q.n_debates)
        rows.append(
            f"<tr>"
            f'<td><a href="questions/{_esc(q.question_key)}.html">{prompt_preview or _esc(q.question_key[:12])}</a></td>'
            f"<td>{_esc(str(q.target or '?'))}</td>"
            f"<td>{q.n_debates}</td>"
            f"<td>{q.steps[0] if q.steps else '?'}&ndash;{q.steps[-1] if q.steps else '?'}</td>"
            f"<td>{q.wrong_winner_rate:.0%}</td>"
            f"<td>{q.draw_rate:.0%}</td>"
            f"<td>{badge}</td>"
            f"</tr>"
        )
    return f"""
    <h2 style="font-family:var(--font-ui);font-size:0.82rem;color:var(--text-2);margin:1rem 0 0.5rem;
        letter-spacing:0.04em;text-transform:uppercase">Questions ({len(questions)})</h2>
    <table class="q-table">
    <thead><tr>
        <th>question</th><th>target</th><th>debates</th><th>steps</th>
        <th>wrong%</th><th>draw%</th><th>score</th>
    </tr></thead>
    <tbody>{"".join(rows)}</tbody>
    </table>
    """


# ---------------------------------------------------------------------------
# Top-level index
# ---------------------------------------------------------------------------


def render_index_html(
    question_index: dict[str, QuestionRecord],
    group_index: dict[str, GroupRecord],
    output_dir: str,
) -> str:
    """Generate the top-level index page."""
    questions_html = render_questions_section(question_index)
    groups_html = render_groups_section(group_index)

    n_questions = len(question_index)
    n_groups = len(group_index)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Debate Analysis — {_esc(output_dir)}</title>
{_CSS}
</head>
<body>
<header class="ep-header">
    <div class="ep-id">debate analysis index</div>
    <div class="ep-meta">
        <span class="meta-pill">{n_questions} questions</span>
        <span class="meta-pill">{n_groups} groups</span>
        <span class="meta-pill dim">{_esc(output_dir)}</span>
    </div>
</header>
<main style="max-width:960px;padding:1rem 2rem;">
{questions_html}
{groups_html}
</main>
{_JS}
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Render question-level and group-level debate pages."
    )
    parser.add_argument(
        "episodes_dir",
        help="Directory containing episodes.jsonl (and optionally groups.jsonl)",
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Output directory (default: <episodes_dir>/html/)",
    )
    args = parser.parse_args()

    episodes_dir = Path(args.episodes_dir)
    episodes_path = episodes_dir / "episodes.jsonl"
    groups_path = episodes_dir / "groups.jsonl"

    if not episodes_path.exists():
        print(f"Error: {episodes_path} not found", file=sys.stderr)
        sys.exit(1)

    # Output dir
    out_dir = Path(args.output_dir) if args.output_dir else episodes_dir / "html"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "questions").mkdir(exist_ok=True)
    (out_dir / "groups").mkdir(exist_ok=True)

    # Load and build indices
    print("Loading episodes...", file=sys.stderr)
    rows = load_rows(episodes_path)
    debates = build_debates(rows)
    print(f"  {len(rows)} rows → {len(debates)} debates", file=sys.stderr)

    question_index = build_question_index(debates)
    print(f"  {len(question_index)} questions", file=sys.stderr)

    group_index: dict[str, object] = {}
    if groups_path.exists():
        group_rows = load_rows(groups_path)
        group_index = build_group_index(group_rows, debates)
        print(f"  {len(group_index)} groups", file=sys.stderr)
    else:
        print("  no groups.jsonl found, skipping group pages", file=sys.stderr)

    # Render question pages
    for qk, qr in question_index.items():
        html = render_question_page(qr)
        path = out_dir / "questions" / f"{qk}.html"
        with open(path, "w") as f:
            f.write(html)
    print(f"Rendered {len(question_index)} question pages", file=sys.stderr)

    # Render group pages
    for gid, gr in group_index.items():
        html = render_group_page(gr)
        path = out_dir / "groups" / f"{gid}.html"
        with open(path, "w") as f:
            f.write(html)
    print(f"Rendered {len(group_index)} group pages", file=sys.stderr)

    # Render top-level index
    index_html = render_index_html(question_index, group_index, str(out_dir))
    index_path = out_dir / "index.html"
    with open(index_path, "w") as f:
        f.write(index_html)
    print(f"\nIndex → {index_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
