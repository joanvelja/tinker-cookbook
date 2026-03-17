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
import html as html_mod
import json
import math
import re
import sys
from pathlib import Path

from scripts.replay_debate import (
    Role,
    _advance_state,
    _empty_state,
    _episode_to_spec,
    _turn_to_utterance,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.core.visibility import build_generation_messages

# ---------------------------------------------------------------------------
# Text parsing helpers
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    return html_mod.escape(text)


def _extract_think(text: str) -> tuple[str | None, str]:
    """Split <think>...</think> or <thinking>...</thinking> from text."""
    m = re.search(r"<(think(?:ing)?)>(.*?)</\1>", text, re.DOTALL)
    if not m:
        return None, text
    thinking = m.group(2).strip()
    rest = (text[: m.start()] + text[m.end() :]).strip()
    return (thinking if thinking else None), rest


def _extract_answer(text: str) -> str | None:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _render_output_html(text: str, role: str) -> str:
    """Render model output with thinking blocks and answer tags styled.

    Emits both a raw <pre class="output-text"> (hidden by default) and a
    <div class="output-rendered"> that JS will process with marked + KaTeX.
    """
    thinking, rest = _extract_think(text)
    answer = _extract_answer(rest)
    parts = []

    if thinking:
        parts.append(
            f'<details class="think-block">'
            f'<summary class="think-summary">reasoning</summary>'
            f'<pre class="think-content">{_esc(thinking)}</pre>'
            f"</details>"
        )

    # Raw pre (hidden by default — rendered mode is default)
    if answer:
        highlighted = re.sub(
            r"<answer>(.*?)</answer>",
            lambda m: f'<span class="answer-tag">&lt;answer&gt;{_esc(m.group(1))}&lt;/answer&gt;</span>',
            _esc(rest),
            flags=re.DOTALL,
        )
        parts.append(f'<pre class="output-text" style="display:none">{highlighted}</pre>')
    else:
        parts.append(f'<pre class="output-text" style="display:none">{_esc(rest)}</pre>')

    # Rendered div (visible by default — JS will process on load)
    parts.append(f'<div class="output-rendered">{_esc(rest)}</div>')

    if answer:
        parts.append(f'<div class="answer-extracted">answer: {_esc(answer)}</div>')

    return "\n".join(parts)


def _render_messages_html(messages: list[dict]) -> str:
    """Render a list of chat messages as styled HTML."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f'<div class="msg msg--{role}">')
        parts.append(f'<div class="msg-role">{_esc(role)}</div>')
        parts.append(f'<pre class="msg-content">{_esc(content)}</pre>')
        parts.append("</div>")
    return "\n".join(parts)


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


def _build_oracle_perspective(ep: dict) -> list[dict]:
    """Build oracle view: for each turn, show the generating debater's full I/O."""
    turns_a = _build_debater_perspective(ep, Role.DEBATER_A)
    turns_b = _build_debater_perspective(ep, Role.DEBATER_B)

    a_by_idx = {t["index"]: t for t in turns_a if t["is_mine"]}
    b_by_idx = {t["index"]: t for t in turns_b if t["is_mine"]}

    oracle_turns = []
    for i in range(len(ep["transcript"])):
        if i in a_by_idx:
            oracle_turns.append(a_by_idx[i])
        elif i in b_by_idx:
            oracle_turns.append(b_by_idx[i])
    return oracle_turns


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
# Signals panel helper
# ---------------------------------------------------------------------------


def _render_signals_html(ep: dict) -> str:
    """Render collapsible signals/grader panel."""
    signals = ep.get("signals", {})
    if not signals:
        return ""

    # Group by prefix
    grouped: dict[str, list[tuple[str, object]]] = {}
    for k, v in sorted(signals.items()):
        prefix = k.split(".")[0] if "." in k else ""
        grouped.setdefault(prefix, []).append((k, v))

    n = len(signals)
    parts = [f'<details class="signals-panel">']
    parts.append(f'<summary>signals ({n} keys)</summary>')
    parts.append('<div class="signals-grid">')

    for prefix, items in grouped.items():
        if prefix:
            parts.append(f'<div class="signals-group-label">{_esc(prefix)}.*</div>')
            parts.append('<div class="signals-group-spacer"></div>')
        for k, v in items:
            if isinstance(v, float):
                if math.isnan(v):
                    v_str = "NaN"
                elif math.isinf(v):
                    v_str = "Inf" if v > 0 else "-Inf"
                else:
                    v_str = f"{v:.3f}"
            else:
                v_str = str(v)
            parts.append(f'<div class="signals-key">{_esc(k)}</div>')
            parts.append(f'<div class="signals-val">{_esc(v_str)}</div>')

    parts.append("</div>")
    parts.append("</details>")
    return "\n".join(parts)


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
    parts.append(_render_output_html(turn["text"], role))
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
        parts.append(_render_output_html(judge["verdict_text"], "judge"))

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
    oracle_turns = _build_oracle_perspective(ep)
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
# CSS
# ---------------------------------------------------------------------------

_CSS = """<style>
:root {
    --bg: #0a0a0e;
    --surface: #111118;
    --surface-2: #1a1a24;
    --surface-3: #222230;
    --border: #2a2a38;
    --border-sub: #1c1c28;

    --text-1: #d2d0cc;
    --text-2: #8a8894;
    --text-3: #55535e;

    --role-a: #4ade80;
    --role-a-dim: #16653480;
    --role-a-bg: rgba(74, 222, 128, 0.05);

    --role-b: #f472b6;
    --role-b-dim: #83184380;
    --role-b-bg: rgba(244, 114, 182, 0.05);

    --role-judge: #60a5fa;
    --role-judge-dim: #1e3a5f80;
    --role-judge-bg: rgba(96, 165, 250, 0.05);

    --role-oracle: #fbbf24;
    --role-oracle-dim: #78590580;
    --role-oracle-bg: rgba(251, 191, 36, 0.05);

    --accent: #fbbf24;
    --correct: #4ade80;
    --incorrect: #f87171;

    --font-mono: ui-monospace, "Cascadia Code", "SF Mono", Menlo, Consolas, monospace;
    --font-body: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, "Times New Roman", serif;
    --font-ui: system-ui, -apple-system, "Segoe UI", sans-serif;

    --radius: 6px;
}

/* Light theme overrides */
[data-theme="light"] {
    --bg: #f5f5f0;
    --surface: #ffffff;
    --surface-2: #f0eff0;
    --surface-3: #e5e4e8;
    --border: #d4d3d8;
    --border-sub: #e8e7ec;

    --text-1: #1a1a1e;
    --text-2: #5a5860;
    --text-3: #9a98a0;

    --role-a-bg: rgba(74, 222, 128, 0.08);
    --role-b-bg: rgba(244, 114, 182, 0.08);
    --role-judge-bg: rgba(96, 165, 250, 0.08);
    --role-oracle-bg: rgba(251, 191, 36, 0.08);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html { font-size: 15px; scroll-behavior: smooth; }
body {
    background: var(--bg);
    color: var(--text-1);
    font-family: var(--font-mono);
    font-size: 0.82rem;
    line-height: 1.6;
    min-height: 100vh;
    padding: 0;
}

/* --- Header --- */
.ep-header {
    padding: 1.5rem 2rem 1rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
}
.ep-id {
    font-family: var(--font-ui);
    font-size: 0.72rem;
    color: var(--text-3);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.ep-id-hash { color: var(--text-3); font-family: var(--font-mono); font-size: 0.68rem; }
.ep-question {
    font-family: var(--font-body);
    font-size: 1.05rem;
    line-height: 1.55;
    color: var(--text-1);
    margin-bottom: 0.75rem;
    max-width: 72ch;
}
.ep-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    align-items: center;
}
.meta-pill {
    font-size: 0.72rem;
    padding: 0.15rem 0.55rem;
    background: var(--surface-2);
    border: 1px solid var(--border-sub);
    border-radius: 99px;
    color: var(--text-2);
    white-space: nowrap;
}
.meta-pill b { color: var(--text-1); font-weight: 600; }
.meta-pill.dim { color: var(--text-3); }
.meta-pill.winner--a { border-color: var(--role-a-dim); }
.meta-pill.winner--a b { color: var(--role-a); }
.meta-pill.winner--b { border-color: var(--role-b-dim); }
.meta-pill.winner--b b { color: var(--role-b); }
.meta-pill.winner--tie b { color: var(--text-2); }

/* --- Signals panel --- */
.signals-panel {
    margin-top: 0.6rem;
    font-family: var(--font-mono);
    font-size: 0.72rem;
}
.signals-panel > summary {
    font-family: var(--font-ui);
    font-size: 0.72rem;
    color: var(--text-3);
    cursor: pointer;
    user-select: none;
    letter-spacing: 0.03em;
    padding: 0.2rem 0;
}
.signals-panel > summary:hover { color: var(--text-2); }
.signals-grid {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 0.1rem 1rem;
    padding: 0.4rem 0;
    max-width: 50rem;
}
.signals-key { color: var(--text-3); }
.signals-val { color: var(--text-1); font-weight: 500; }
.signals-group-label {
    color: var(--accent);
    font-family: var(--font-ui);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding-top: 0.35rem;
    grid-column: 1 / -1;
}
.signals-group-spacer { display: none; }

/* --- Tabs --- */
.tabs {
    display: flex;
    align-items: center;
    gap: 0;
    padding: 0 2rem;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 10;
}
.tab {
    font-family: var(--font-ui);
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    padding: 0.65rem 1.2rem;
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--text-3);
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    transition: color 0.15s, border-color 0.15s;
}
.tab:hover { color: var(--text-2); }
.tab.active { color: var(--text-1); }
.tab--a.active { border-bottom-color: var(--role-a); }
.tab--b.active { border-bottom-color: var(--role-b); }
.tab--judge.active { border-bottom-color: var(--role-judge); }
.tab--oracle.active { border-bottom-color: var(--accent); }

.tab-pip {
    width: 7px; height: 7px;
    border-radius: 50%;
    display: inline-block;
}
.tab-pip[data-role="debater_a"], [data-role="debater_a"].role-pip { background: var(--role-a); }
.tab-pip[data-role="debater_b"], [data-role="debater_b"].role-pip { background: var(--role-b); }
.tab-pip[data-role="judge"], [data-role="judge"].role-pip { background: var(--role-judge); }
.tab-pip[data-role="oracle"] { background: var(--accent); }

.tab-actions {
    margin-left: auto;
    display: flex;
    gap: 0.3rem;
}
.btn-action {
    font-family: var(--font-ui);
    font-size: 0.68rem;
    padding: 0.25rem 0.5rem;
    background: var(--surface-2);
    border: 1px solid var(--border-sub);
    border-radius: 4px;
    color: var(--text-3);
    cursor: pointer;
}
.btn-action:hover { color: var(--text-2); background: var(--surface-3); }

/* --- Outline strip --- */
.outline {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.4rem 2rem;
    background: var(--surface);
    border-bottom: 1px solid var(--border-sub);
    overflow-x: auto;
    white-space: nowrap;
    position: sticky;
    top: 38px;  /* below tabs */
    z-index: 9;
}
.outline-chip {
    font-family: var(--font-ui);
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    padding: 0.15rem 0.45rem;
    border-radius: 99px;
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-3);
    cursor: pointer;
    white-space: nowrap;
    transition: color 0.12s, border-color 0.12s;
}
.outline-chip:hover { color: var(--text-1); }
.outline-chip[data-role="debater_a"] { border-color: var(--role-a-dim); color: var(--role-a); }
.outline-chip[data-role="debater_a"]:hover { border-color: var(--role-a); }
.outline-chip[data-role="debater_b"] { border-color: var(--role-b-dim); color: var(--role-b); }
.outline-chip[data-role="debater_b"]:hover { border-color: var(--role-b); }
.outline-chip[data-role="judge"] { border-color: var(--role-judge-dim); color: var(--role-judge); }
.outline-chip[data-role="judge"]:hover { border-color: var(--role-judge); }
.outline-arrow {
    font-size: 0.6rem;
    color: var(--text-3);
    opacity: 0.5;
}

/* --- Perspectives --- */
.perspective { display: none; padding: 1rem 2rem 3rem; }
.perspective.active { display: block; }

main { max-width: 960px; }

/* --- Turns --- */
.turns { display: flex; flex-direction: column; gap: 0.75rem; }

.turn {
    border: 1px solid var(--border-sub);
    border-radius: var(--radius);
    background: var(--surface);
    overflow: hidden;
}
.turn--mine { border-left: 3px solid var(--border); }
.turn--other { opacity: 0.55; }
.turn--other:hover { opacity: 0.85; }
.turn--mine[data-role="debater_a"] { border-left-color: var(--role-a); background: var(--role-a-bg); }
.turn--mine[data-role="debater_b"] { border-left-color: var(--role-b); background: var(--role-b-bg); }

/* Turn header */
.turn-hdr {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.45rem 0.75rem;
    border-bottom: 1px solid var(--border-sub);
    background: rgba(255,255,255,0.015);
}
[data-theme="light"] .turn-hdr { background: rgba(0,0,0,0.02); }
.turn-idx {
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--text-3);
    min-width: 1.8rem;
}
.role-pip {
    width: 6px; height: 6px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}
.turn-role {
    font-size: 0.72rem;
    font-weight: 600;
}
.turn-role[data-role="debater_a"] { color: var(--role-a); }
.turn-role[data-role="debater_b"] { color: var(--role-b); }
.turn-role[data-role="judge"] { color: var(--role-judge); }
.turn-phase, .turn-round {
    font-size: 0.68rem;
    color: var(--text-3);
}
.turn-badge-mine {
    font-size: 0.62rem;
    padding: 0.08rem 0.4rem;
    border-radius: 99px;
    background: rgba(255,255,255,0.06);
    color: var(--text-3);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-left: auto;
}
[data-theme="light"] .turn-badge-mine { background: rgba(0,0,0,0.05); }

/* Input section */
.turn-input {
    border-bottom: 1px solid var(--border-sub);
}
.input-summary {
    font-family: var(--font-ui);
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--text-3);
    padding: 0.4rem 0.75rem;
    cursor: pointer;
    user-select: none;
    letter-spacing: 0.02em;
}
.input-summary:hover { color: var(--text-2); }
.input-summary::marker { color: var(--text-3); }
.input-body {
    padding: 0.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    background: rgba(0,0,0,0.2);
}
[data-theme="light"] .input-body { background: rgba(0,0,0,0.03); }

/* Messages */
.msg {
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid var(--border-sub);
}
.msg-role {
    font-family: var(--font-ui);
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0.2rem 0.55rem;
    background: rgba(255,255,255,0.03);
    border-bottom: 1px solid var(--border-sub);
}
[data-theme="light"] .msg-role { background: rgba(0,0,0,0.02); }
.msg--system .msg-role { color: var(--accent); }
.msg--user .msg-role { color: var(--text-2); }
.msg--assistant .msg-role { color: #a78bfa; }
.msg--prefill .msg-role { color: var(--text-3); font-style: italic; }
.msg-content {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    line-height: 1.55;
    padding: 0.55rem;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 400px;
    overflow-y: auto;
    color: var(--text-2);
    background: var(--bg);
}

/* Output section */
.turn-output {
    padding: 0.65rem 0.75rem;
}
.output-text {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--text-1);
    max-height: 600px;
    overflow-y: auto;
}

/* Rendered markdown output */
.output-rendered {
    font-family: var(--font-ui);
    font-size: 0.88rem;
    line-height: 1.7;
    color: var(--text-1);
    max-height: 600px;
    overflow-y: auto;
}
.output-rendered h1 { font-size: 1.3rem; font-weight: 700; margin: 0.8rem 0 0.4rem; font-family: var(--font-ui); }
.output-rendered h2 { font-size: 1.1rem; font-weight: 600; margin: 0.7rem 0 0.3rem; font-family: var(--font-ui); }
.output-rendered h3 { font-size: 0.95rem; font-weight: 600; margin: 0.5rem 0 0.2rem; font-family: var(--font-ui); }
.output-rendered h4, .output-rendered h5, .output-rendered h6 {
    font-size: 0.88rem; font-weight: 600; margin: 0.4rem 0 0.2rem; font-family: var(--font-ui);
}
.output-rendered p { margin: 0.4rem 0; }
.output-rendered ul, .output-rendered ol {
    margin: 0.3rem 0; padding-left: 1.5rem;
}
.output-rendered li { margin: 0.15rem 0; }
.output-rendered code {
    font-family: var(--font-mono); font-size: 0.82em;
    background: var(--surface-2); border: 1px solid var(--border-sub);
    border-radius: 3px; padding: 0.1rem 0.3rem;
}
.output-rendered pre {
    background: var(--surface-2); border: 1px solid var(--border-sub);
    border-radius: var(--radius); padding: 0.6rem; margin: 0.4rem 0;
    overflow-x: auto;
}
.output-rendered pre code {
    background: none; border: none; padding: 0;
    font-size: 0.78rem; line-height: 1.5;
}
.output-rendered blockquote {
    border-left: 3px solid var(--border); padding: 0.3rem 0.75rem;
    margin: 0.4rem 0; color: var(--text-2); font-style: italic;
}
.output-rendered table {
    border-collapse: collapse; margin: 0.4rem 0; font-size: 0.82rem;
    font-family: var(--font-mono);
}
.output-rendered th, .output-rendered td {
    border: 1px solid var(--border-sub); padding: 0.3rem 0.5rem;
}
.output-rendered th { background: var(--surface-2); font-weight: 600; }
.output-rendered a { color: var(--role-judge); text-decoration: underline; }
.output-rendered hr { border: none; border-top: 1px solid var(--border); margin: 0.6rem 0; }

/* Thinking blocks */
.think-block {
    margin-bottom: 0.5rem;
    border: 1px solid var(--border-sub);
    border-radius: 4px;
    background: rgba(251, 191, 36, 0.03);
}
.think-summary {
    font-family: var(--font-ui);
    font-size: 0.68rem;
    color: var(--accent);
    padding: 0.3rem 0.55rem;
    cursor: pointer;
    letter-spacing: 0.03em;
    opacity: 0.7;
}
.think-summary:hover { opacity: 1; }
.think-content {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    line-height: 1.5;
    padding: 0.55rem;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--text-3);
    max-height: 400px;
    overflow-y: auto;
    border-top: 1px solid var(--border-sub);
}

/* Answer tags */
.answer-tag {
    background: rgba(74, 222, 128, 0.1);
    color: var(--role-a);
    border-radius: 3px;
    padding: 0 0.2rem;
}
.answer-extracted {
    margin-top: 0.5rem;
    font-family: var(--font-ui);
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--accent);
    padding: 0.25rem 0.55rem;
    background: rgba(251, 191, 36, 0.06);
    border: 1px solid rgba(251, 191, 36, 0.15);
    border-radius: 4px;
    display: inline-block;
}

/* Judge block */
.judge-block {
    margin-top: 1rem;
    border: 1px solid var(--role-judge-dim);
    border-radius: var(--radius);
    background: var(--role-judge-bg);
    overflow: hidden;
}
.judge-label {
    font-family: var(--font-ui);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--role-judge);
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--role-judge-dim);
}
.judge-section { padding: 0.5rem 0.75rem; }
.judge-verdict-hdr {
    font-family: var(--font-ui);
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text-1);
    margin: 0.5rem 0;
    padding: 0.3rem 0;
    border-bottom: 1px solid var(--border-sub);
}

/* Scrollbars */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--surface-3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-3); }

/* Utility */
.dim { color: var(--text-3) !important; }
</style>"""


# ---------------------------------------------------------------------------
# JS
# ---------------------------------------------------------------------------

_JS = """<script>
// --- Theme toggle ---
(function initTheme() {
    const saved = localStorage.getItem('debate-viewer-theme');
    if (saved) document.documentElement.dataset.theme = saved;
    updateThemeBtn();
})();

function toggleTheme() {
    const html = document.documentElement;
    const next = html.dataset.theme === 'light' ? 'dark' : 'light';
    html.dataset.theme = next;
    localStorage.setItem('debate-viewer-theme', next);
    updateThemeBtn();
}

function updateThemeBtn() {
    const btn = document.getElementById('btn-theme');
    if (!btn) return;
    btn.textContent = document.documentElement.dataset.theme === 'light' ? '\\u2600' : '\\u263e';
}

// --- Tab switching ---
document.querySelectorAll('.tab[data-perspective]').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => {
            t.classList.remove('active');
            t.setAttribute('aria-selected', 'false');
        });
        tab.classList.add('active');
        tab.setAttribute('aria-selected', 'true');

        const p = tab.dataset.perspective;
        document.querySelectorAll('.perspective').forEach(el => el.classList.remove('active'));
        document.getElementById('p-' + p).classList.add('active');
    });
});

// --- Keyboard nav ---
document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    const tabs = [...document.querySelectorAll('.tab[data-perspective]')];
    const active = tabs.findIndex(t => t.classList.contains('active'));
    if (e.key === 'ArrowLeft' && active > 0) tabs[active - 1].click();
    if (e.key === 'ArrowRight' && active < tabs.length - 1) tabs[active + 1].click();
});

// --- Expand/collapse all ---
function toggleAll(open) {
    const perspective = document.querySelector('.perspective.active');
    if (!perspective) return;
    perspective.querySelectorAll('details').forEach(d => d.open = open);
}

// --- Outline scroll ---
function scrollToTurn(index) {
    // Find active perspective
    const active = document.querySelector('.perspective.active');
    if (!active) return;
    const perspId = active.id.replace('p-', '');  // 'a', 'b', 'judge', 'oracle'
    const el = document.getElementById('turn-' + perspId + '-' + index);
    if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

function scrollToJudge() {
    const active = document.querySelector('.perspective.active');
    if (!active) return;
    const perspId = active.id.replace('p-', '');
    const el = document.getElementById('judge-' + perspId);
    if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// --- Render mode toggle (raw <pre> vs rendered markdown) ---
var renderMode = 'rendered';

function toggleRenderMode() {
    renderMode = renderMode === 'rendered' ? 'raw' : 'rendered';
    applyRenderMode();
    const btn = document.getElementById('btn-render-toggle');
    if (btn) btn.textContent = renderMode === 'rendered' ? 'Rendered' : 'Raw';
}

function applyRenderMode() {
    document.querySelectorAll('.output-text').forEach(el => {
        el.style.display = renderMode === 'raw' ? '' : 'none';
    });
    document.querySelectorAll('.output-rendered').forEach(el => {
        el.style.display = renderMode === 'rendered' ? '' : 'none';
    });
}

// --- Markdown + KaTeX rendering on load ---
function initRendering() {
    if (typeof marked === 'undefined') {
        // marked not loaded yet, retry
        setTimeout(initRendering, 100);
        return;
    }
    document.querySelectorAll('.output-rendered').forEach(el => {
        let raw = el.textContent;
        // Escape non-standard XML tags so marked doesn't strip them
        raw = raw.replace(/<(\/?)(?!(?:p|br|hr|div|span|a|b|i|u|em|strong|code|pre|ul|ol|li|h[1-6]|table|thead|tbody|tr|th|td|blockquote|img|details|summary|sub|sup|del|ins|mark|small|s|abbr|cite|dfn|kbd|samp|var|dl|dt|dd|figure|figcaption|caption|col|colgroup|section|article|header|footer|nav|aside|main|ruby|rt|rp|bdi|bdo|wbr|time|data|output|progress|meter|source|track|video|audio|picture|map|area|canvas|svg|math)\b)(\w[\w-]*)([^>]*)>/g,
            (m, slash, tag, attrs) => '&lt;' + slash + tag + attrs + '&gt;');
        el.innerHTML = marked.parse(raw);
        if (typeof renderMathInElement !== 'undefined') {
            renderMathInElement(el, {
                delimiters: [
                    {left: "$$", right: "$$", display: true},
                    {left: "$", right: "$", display: false},
                    {left: "\\\\(", right: "\\\\)", display: false},
                    {left: "\\\\[", right: "\\\\]", display: true},
                ],
                throwOnError: false
            });
        }
    });
    applyRenderMode();
}

// Wait for all deferred scripts
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => setTimeout(initRendering, 50));
} else {
    setTimeout(initRendering, 50);
}
</script>"""


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
        out_dir = Path(args.episodes_path).parent / "html"

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
