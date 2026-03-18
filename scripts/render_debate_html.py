#!/usr/bin/env python3
"""Render debate episodes as rich self-contained HTML files.

Generates per-episode HTML with four togglable perspectives (Debater A,
Debater B, Judge, Oracle) showing exact model I/O at each turn.

Features:
  - Dark/Light mode toggle (persisted to localStorage)
  - Turn outline strip (clickable, sticky)
  - Signals/Grader panel (collapsible, from episode signals)
  - Markdown + KaTeX math rendering (toggle raw/rendered)
  - Oracle tab (chronological full-I/O from generating debater)

Usage:
    # Render all episodes
    uv run python scripts/render_debate_html.py logs/.../episodes/episodes.jsonl

    # Render specific episodes
    uv run python scripts/render_debate_html.py logs/.../episodes/episodes.jsonl -i 0,1,5

    # Custom output dir
    uv run python scripts/render_debate_html.py logs/.../episodes/episodes.jsonl -o /tmp/debate-viewer
"""

import argparse
import json
import sys
from pathlib import Path

from scripts.debate_style import (
    _CSS,
    _JS,
    _esc,
    _render_messages_html,
    _render_output_html,
    _render_signals_html,
)
from scripts.replay_debate import (
    Role,
    _advance_state,
    _empty_state,
    _episode_to_spec,
    _turn_to_utterance,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.core.visibility import build_generation_messages


# ---------------------------------------------------------------------------
# Perspective building
# ---------------------------------------------------------------------------


def _build_debater_perspective(ep: dict, viewer: Role) -> list[dict]:
    """Build turn-by-turn I/O from a debater's perspective."""
    spec = _episode_to_spec(ep)
    state = _empty_state(spec)
    utterances = [_turn_to_utterance(t, spec.schedule) for t in ep["transcript"]]
    turns = []

    for i, utt in enumerate(utterances):
        turn = {
            "index": i,
            "round": utt.round_index,
            "phase": utt.phase.value,
            "role": utt.role.value,
            "is_mine": utt.role == viewer,
            "text": ep["transcript"][i]["text"],
        }

        if utt.role == viewer:
            msgs, prefill = build_generation_messages(state, viewer)
            turn["input_messages"] = [
                {"role": m["role"], "content": m["content"]} for m in msgs
            ]
            turn["prefill"] = prefill

        turns.append(turn)
        state = _advance_state(state, utt)

    return turns


def _build_oracle_from_perspectives(
    turns_a: list[dict], turns_b: list[dict], n_turns: int
) -> list[dict]:
    """Derive oracle view from pre-computed A and B perspectives (no extra replay)."""
    a_by_idx = {t["index"]: t for t in turns_a if t["is_mine"]}
    b_by_idx = {t["index"]: t for t in turns_b if t["is_mine"]}
    return [a_by_idx.get(i) or b_by_idx.get(i) for i in range(n_turns) if i in a_by_idx or i in b_by_idx]


def _build_judge_perspective(ep: dict) -> dict:
    """Build judge's view: full assembled input + verdict."""
    spec = _episode_to_spec(ep)
    state = _empty_state(spec)
    utterances = [_turn_to_utterance(t, spec.schedule) for t in ep["transcript"]]

    for utt in utterances:
        state = _advance_state(state, utt)

    msgs, prefill = build_generation_messages(state, Role.JUDGE, trigger="final")
    return {
        "input_messages": [{"role": m["role"], "content": m["content"]} for m in msgs],
        "prefill": prefill,
        "verdict_text": ep.get("verdict_text") or "",
        "winner": ep.get("winner"),
    }


# ---------------------------------------------------------------------------
# Outline strip helper
# ---------------------------------------------------------------------------


def _render_outline_html(ep: dict) -> str:
    """Render the horizontal turn outline strip."""
    transcript = ep.get("transcript", [])
    if not transcript:
        return ""

    spec = _episode_to_spec(ep)
    utterances = [_turn_to_utterance(t, spec.schedule) for t in transcript]

    parts = ['<div class="outline">']
    for i, utt in enumerate(utterances):
        role = utt.role.value
        phase = utt.phase.value
        # Short label: "A propose" or "B cross"
        role_short = "A" if role == "debater_a" else ("B" if role == "debater_b" else "J")
        label = f"T{i} {role_short} {phase}"
        parts.append(
            f'<button class="outline-chip" data-role="{role}" '
            f'onclick="scrollToTurn({i})">{_esc(label)}</button>'
        )
        if i < len(utterances) - 1:
            parts.append('<span class="outline-arrow">\u2192</span>')

    # Judge chip at the end
    parts.append('<span class="outline-arrow">\u2192</span>')
    parts.append(
        '<button class="outline-chip" data-role="judge" '
        'onclick="scrollToJudge()">Judge</button>'
    )
    parts.append("</div>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def _render_turn_html(turn: dict, perspective: str) -> str:
    """Render a single turn card."""
    role = turn["role"]
    mine_cls = "turn--mine" if turn.get("is_mine") else "turn--other"
    phase_label = turn["phase"]
    round_label = f'R{turn["round"]}'
    turn_id = f'turn-{perspective}-{turn["index"]}'

    parts = [f'<div class="turn {mine_cls}" data-role="{role}" id="{turn_id}">']

    # Header
    parts.append('<div class="turn-hdr">')
    parts.append(f'<span class="turn-idx">T{turn["index"]}</span>')
    parts.append(f'<span class="role-pip" data-role="{role}"></span>')
    parts.append(f'<span class="turn-role" data-role="{role}">{_esc(role)}</span>')
    parts.append(f'<span class="turn-phase">{_esc(phase_label)}</span>')
    parts.append(f'<span class="turn-round">{_esc(round_label)}</span>')
    if turn.get("is_mine"):
        parts.append('<span class="turn-badge-mine">generating</span>')
    parts.append("</div>")

    # Input (only for "mine" turns)
    if turn.get("input_messages"):
        n = len(turn["input_messages"])
        parts.append(f'<details class="turn-input">')
        parts.append(
            f'<summary class="input-summary">'
            f"input ({n} message{'s' if n != 1 else ''})</summary>"
        )
        parts.append(f'<div class="input-body">')
        parts.append(_render_messages_html(turn["input_messages"]))
        if turn.get("prefill"):
            parts.append(f'<div class="msg msg--prefill">')
            parts.append(f'<div class="msg-role">prefill</div>')
            parts.append(f'<pre class="msg-content">{_esc(turn["prefill"])}</pre>')
            parts.append("</div>")
        parts.append("</div>")
        parts.append("</details>")

    # Output
    parts.append('<div class="turn-output">')
    parts.append(_render_output_html(turn["text"]))
    parts.append("</div>")

    parts.append("</div>")
    return "\n".join(parts)


def _render_judge_html(judge: dict) -> str:
    """Render judge perspective."""
    parts = ['<div class="judge-section">']

    # Judge input
    n = len(judge["input_messages"])
    parts.append('<details class="turn-input">')
    parts.append(
        f'<summary class="input-summary">'
        f"judge input ({n} message{'s' if n != 1 else ''})</summary>"
    )
    parts.append('<div class="input-body">')
    parts.append(_render_messages_html(judge["input_messages"]))
    parts.append("</div>")
    parts.append("</details>")

    # Verdict
    winner = judge["winner"] or "tie"
    parts.append(f'<div class="judge-verdict-hdr">verdict: {_esc(str(winner))}</div>')
    if judge["verdict_text"]:
        parts.append(_render_output_html(judge["verdict_text"]))

    parts.append("</div>")
    return "\n".join(parts)


def _render_perspective_html(turns: list[dict], judge: dict, perspective: str) -> str:
    """Render a full perspective panel (turns + judge)."""
    parts = ['<div class="turns">']
    for turn in turns:
        parts.append(_render_turn_html(turn, perspective))
    parts.append("</div>")
    parts.append(f'<div class="judge-block" data-role="judge" id="judge-{perspective}">')
    parts.append('<div class="judge-label">Judge</div>')
    parts.append(_render_judge_html(judge))
    parts.append("</div>")
    return "\n".join(parts)


def render_episode_html(ep: dict, index: int) -> str:
    """Generate self-contained HTML for one debate episode."""
    # Build perspectives
    turns_a = _build_debater_perspective(ep, Role.DEBATER_A)
    turns_b = _build_debater_perspective(ep, Role.DEBATER_B)
    oracle_turns = _build_oracle_from_perspectives(turns_a, turns_b, len(ep.get("transcript", [])))
    judge = _build_judge_perspective(ep)

    # Metadata
    target = ep.get("target", "?")
    winner = ep.get("winner") or "tie"
    reward = ep.get("reward", 0.0)
    answers = ep.get("answers", {})
    protocol = ep.get("protocol_kind", "?")
    step = ep.get("step", "?")
    split = ep.get("split", "?")
    prompts_ref = ep.get("prompts_ref", "?")
    task_prompt = ep.get("task_prompt", "")
    answer_a = answers.get("public_debater_a", "?")
    answer_b = answers.get("public_debater_b", "?")
    n_turns = len(ep.get("transcript", []))
    debate_id = ep.get("debate_id", "?")[:12]

    # Winner color class
    winner_cls = {"debater_a": "winner--a", "debater_b": "winner--b"}.get(
        str(winner), "winner--tie"
    )

    signals_html = _render_signals_html(ep)
    outline_html = _render_outline_html(ep)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Debate #{index} — {_esc(str(target)[:60])}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/contrib/auto-render.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/marked@15/marked.min.js"></script>
{_CSS}
</head>
<body>

<header class="ep-header">
    <div class="ep-id">episode {index} <span class="ep-id-hash">{_esc(debate_id)}</span></div>
    <div class="ep-question">{_esc(task_prompt) if task_prompt else '<span class="dim">task_prompt not available (older episode format)</span>'}</div>
    <div class="ep-meta">
        <span class="meta-pill">target: <b>{_esc(str(target))}</b></span>
        <span class="meta-pill {winner_cls}">verdict: <b>{_esc(str(winner))}</b></span>
        <span class="meta-pill">reward: <b>{reward:+.2f}</b></span>
        <span class="meta-pill">A: <b>{_esc(str(answer_a)[:40])}</b></span>
        <span class="meta-pill">B: <b>{_esc(str(answer_b)[:40])}</b></span>
        <span class="meta-pill dim">{_esc(protocol)} · step {step} · {split} · {n_turns} turns · {_esc(prompts_ref)}</span>
    </div>
    {signals_html}
</header>

<nav class="tabs" role="tablist">
    <button class="tab tab--a active" data-perspective="a" role="tab" aria-selected="true">
        <span class="tab-pip" data-role="debater_a"></span>Debater A
    </button>
    <button class="tab tab--b" data-perspective="b" role="tab" aria-selected="false">
        <span class="tab-pip" data-role="debater_b"></span>Debater B
    </button>
    <button class="tab tab--judge" data-perspective="judge" role="tab" aria-selected="false">
        <span class="tab-pip" data-role="judge"></span>Judge
    </button>
    <button class="tab tab--oracle" data-perspective="oracle" role="tab" aria-selected="false">
        <span class="tab-pip" data-role="oracle"></span>\u25c8 Oracle
    </button>
    <div class="tab-actions">
        <button class="btn-action" id="btn-render-toggle" onclick="toggleRenderMode()" title="Toggle raw/rendered">Rendered</button>
        <button class="btn-action" onclick="toggleAll(true)" title="Expand all">\u25bc</button>
        <button class="btn-action" onclick="toggleAll(false)" title="Collapse all">\u25b2</button>
        <button class="btn-action" id="btn-theme" onclick="toggleTheme()" title="Toggle dark/light mode">\u263e</button>
    </div>
</nav>

{outline_html}

<main>
    <div class="perspective active" id="p-a">
        {_render_perspective_html(turns_a, judge, "a")}
    </div>
    <div class="perspective" id="p-b">
        {_render_perspective_html(turns_b, judge, "b")}
    </div>
    <div class="perspective" id="p-judge">
        <div class="judge-block" data-role="judge" id="judge-judge">
            <div class="judge-label">Judge — Full View</div>
            {_render_judge_html(judge)}
        </div>
    </div>
    <div class="perspective" id="p-oracle">
        {_render_perspective_html(oracle_turns, judge, "oracle")}
    </div>
</main>

{_JS}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------


def render_index_html(episodes: list[dict], output_dir: str) -> str:
    """Generate an index page linking to individual episode pages."""
    rows = []
    for i, ep in enumerate(episodes):
        winner = ep.get("winner") or "tie"
        winner_cls = {"debater_a": "winner--a", "debater_b": "winner--b"}.get(
            str(winner), "winner--tie"
        )
        rows.append(
            f'<tr>'
            f'<td><a href="episode_{i:04d}.html">#{i}</a></td>'
            f'<td>{ep.get("step", "?")}</td>'
            f'<td>{_esc(ep.get("split", "?"))}</td>'
            f'<td>{_esc(str(ep.get("target", "?"))[:50])}</td>'
            f'<td class="{winner_cls}">{_esc(str(winner))}</td>'
            f'<td>{ep.get("reward", 0.0):+.2f}</td>'
            f'<td>{len(ep.get("transcript", []))}</td>'
            f'<td>{_esc(ep.get("protocol_kind", "?"))}</td>'
            f"</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Debate Episodes — {_esc(output_dir)}</title>
{_CSS}
<style>
.idx-table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
.idx-table th {{ text-align: left; padding: 0.5rem 0.75rem; color: var(--text-2);
    border-bottom: 1px solid var(--border); font-weight: 500;
    font-family: var(--font-ui); letter-spacing: 0.04em; text-transform: uppercase; font-size: 0.7rem; }}
.idx-table td {{ padding: 0.45rem 0.75rem; border-bottom: 1px solid var(--border-sub); }}
.idx-table tr:hover td {{ background: var(--surface-2); }}
.idx-table a {{ color: var(--role-a); text-decoration: none; }}
.idx-table a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<header class="ep-header">
    <div class="ep-id">debate episode index</div>
    <div class="ep-meta">
        <span class="meta-pill">{len(episodes)} episodes</span>
        <span class="meta-pill dim">{_esc(output_dir)}</span>
    </div>
</header>
<main style="padding-top:1rem;">
<table class="idx-table">
<thead><tr>
    <th>#</th><th>step</th><th>split</th><th>target</th>
    <th>verdict</th><th>reward</th><th>turns</th><th>protocol</th>
</tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
</main>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Render debate episodes as HTML.")
    parser.add_argument("episodes_path", help="Path to episodes.jsonl")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory (default: sibling of episodes.jsonl)")
    parser.add_argument("--indices", "-i", default=None, help="Episode indices: 0,1,5 or 0-10")
    args = parser.parse_args()

    # Load episodes
    episodes = []
    with open(args.episodes_path) as f:
        for line in f:
            episodes.append(json.loads(line))

    # Parse indices
    if args.indices:
        if "-" in args.indices and "," not in args.indices:
            lo, hi = args.indices.split("-")
            indices = list(range(int(lo), int(hi) + 1))
        else:
            indices = [int(x) for x in args.indices.split(",")]
    else:
        indices = list(range(len(episodes)))

    # Output dir
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.episodes_path).parent / "html" / "episodes"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Render episodes
    rendered_episodes = []
    for idx in indices:
        if idx >= len(episodes):
            print(f"Warning: index {idx} out of range, skipping", file=sys.stderr)
            continue
        ep = episodes[idx]
        print(f"Rendering episode {idx}...", file=sys.stderr, end=" ", flush=True)
        html_content = render_episode_html(ep, idx)
        out_path = out_dir / f"episode_{idx:04d}.html"
        with open(out_path, "w") as f:
            f.write(html_content)
        print(f"→ {out_path}", file=sys.stderr)
        rendered_episodes.append(ep)

    # Render index
    index_html = render_index_html(
        [episodes[i] for i in indices if i < len(episodes)], str(out_dir)
    )
    index_path = out_dir / "index.html"
    with open(index_path, "w") as f:
        f.write(index_html)
    print(f"\nIndex → {index_path}", file=sys.stderr)
    print(f"Rendered {len(rendered_episodes)} episodes to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
