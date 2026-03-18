"""Shared CSS, JS, and HTML helpers for debate viewers.

Extracted from render_debate_html.py so that render_debate_pages.py
(question + group pages) can reuse the same design system.
"""

import html as html_mod
import math
import re


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

/* --- Question / Group page styles --- */

/* Question table */
.q-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.q-table th {
    text-align: left;
    padding: 0.5rem 0.75rem;
    color: var(--text-2);
    border-bottom: 1px solid var(--border);
    font-weight: 500;
    font-family: var(--font-ui);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-size: 0.7rem;
}
.q-table td {
    padding: 0.45rem 0.75rem;
    border-bottom: 1px solid var(--border-sub);
}
.q-table tr:hover td { background: var(--surface-2); }
.q-table a { color: var(--role-a); text-decoration: none; }
.q-table a:hover { text-decoration: underline; }

/* Group lane */
.g-lane {
    border: 1px solid var(--border-sub);
    border-radius: var(--radius);
    background: var(--surface);
    padding: 0.75rem;
    margin-bottom: 0.5rem;
}
.g-lane-hdr {
    font-family: var(--font-ui);
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-2);
    margin-bottom: 0.4rem;
}

/* Failure banners */
.banner-dead,
.banner-miscredit,
.banner-removed {
    font-family: var(--font-ui);
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.45rem 0.75rem;
    border-radius: var(--radius);
    margin-bottom: 0.5rem;
}
.banner-dead {
    background: rgba(248, 113, 113, 0.08);
    border: 1px solid rgba(248, 113, 113, 0.25);
    color: var(--incorrect);
}
.banner-miscredit {
    background: rgba(251, 191, 36, 0.08);
    border: 1px solid rgba(251, 191, 36, 0.25);
    color: var(--accent);
}
.banner-removed {
    background: rgba(138, 136, 148, 0.08);
    border: 1px solid rgba(138, 136, 148, 0.25);
    color: var(--text-2);
}

/* Reward / advantage bars */
.bar-reward,
.bar-advantage {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    margin: 0.15rem 0;
}
.bar-reward .bar-fill,
.bar-advantage .bar-fill {
    height: 14px;
    border-radius: 2px;
    min-width: 2px;
}
.bar-reward .bar-fill { background: var(--role-judge); }
.bar-advantage .bar-fill--pos { background: var(--correct); }
.bar-advantage .bar-fill--neg { background: var(--incorrect); }
.bar-label {
    min-width: 3.5rem;
    text-align: right;
    color: var(--text-2);
}

/* Step dots */
.step-dots {
    display: flex;
    gap: 3px;
    align-items: center;
    flex-wrap: wrap;
}
.step-dots .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}
.step-dots .dot--correct { background: var(--correct); }
.step-dots .dot--incorrect { background: var(--incorrect); }
.step-dots .dot--draw { background: var(--text-3); }

/* Interestingness badge */
.interestingness-badge {
    font-family: var(--font-ui);
    font-size: 0.65rem;
    font-weight: 700;
    padding: 0.1rem 0.4rem;
    border-radius: 99px;
    display: inline-block;
    letter-spacing: 0.03em;
}
.interestingness-badge--high {
    background: rgba(248, 113, 113, 0.12);
    border: 1px solid rgba(248, 113, 113, 0.3);
    color: var(--incorrect);
}
.interestingness-badge--mid {
    background: rgba(251, 191, 36, 0.12);
    border: 1px solid rgba(251, 191, 36, 0.3);
    color: var(--accent);
}
.interestingness-badge--low {
    background: rgba(138, 136, 148, 0.08);
    border: 1px solid rgba(138, 136, 148, 0.2);
    color: var(--text-3);
}
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
