"""HTML trace formatter for debate rollouts.

Provides rich, role-colored HTML rendering for debate traces via logtree.
Uses the Formatter protocol to inject CSS and log_html() for custom content.

Usage:
    with logtree.init_trace("Debate", path="trace.html"):
        logtree.log_formatter(DebateTraceCSSInjector())  # inject CSS once
        ...
        logtree.log_html(render_rollout_html(env, reward))
"""

from __future__ import annotations

import html as html_module
from dataclasses import replace

from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.utils.logtree import Formatter

from ..prompts import resolve_prompts
from ..types import Role
from ..core.visibility import build_generation_messages


def _esc(text: str) -> str:
    return html_module.escape(text)


def _format_messages_text(msgs: tuple) -> str:
    """Format visible messages as plain text with role headers."""
    parts = []
    for m in msgs:
        role = m.get("role", "???")
        content = get_text_content(m)
        parts.append(f"[{role}]\n{content}")
    return "\n\n---\n\n".join(parts) if parts else "(no messages)"


def render_rollout_html(env, reward: float) -> str:
    """Build full HTML for one debate rollout."""
    state = env.runtime.state
    spec = state.spec
    prompts = resolve_prompts(spec.prompts_ref)

    outcome = state.outcome
    winner_str = outcome.winner.value if outcome and outcome.winner else "tie"
    we_won = (outcome.winner == env.role) if outcome and outcome.winner else None
    result = {True: "win", False: "loss", None: "tie"}[we_won]
    result_label = result.upper()

    total_tokens = sum(u.token_count for u in state.transcript)
    n_turns = len(state.transcript)

    p = []  # parts accumulator

    p.append(f'<div class="db-rollout" data-result="{result}">')

    # -- Summary badges --
    p.append('<div class="db-summary">')
    p.append(f'<span class="db-badge db-role" data-role="{env.role.value}">{_esc(env.role.value)}</span>')
    p.append('<span class="db-badge db-trained">TRAINED</span>')
    p.append(f'<span class="db-badge db-{result}">{result_label} {reward:+.1f}</span>')
    p.append(f'<span class="db-meta">{n_turns} turns \u00b7 {total_tokens} tok \u00b7 verdict: {_esc(winner_str)}</span>')
    p.append("</div>")

    # -- System prompts (collapsed) --
    p.append('<details class="db-sysprompts">')
    p.append("<summary>System Prompts</summary>")
    for role in [Role.DEBATER_A, Role.DEBATER_B, Role.JUDGE]:
        system_text = prompts.render_system(state, role)
        is_ours = role == env.role
        tag = ' <span class="db-badge db-trained">TRAINED</span>' if is_ours else ""
        p.append('<div class="db-sysprompt">')
        p.append(f'<div class="db-sysprompt-hdr" data-role="{role.value}">{_esc(role.value)}{tag}</div>')
        p.append(f'<pre class="db-pre">{_esc(system_text)}</pre>')
        p.append("</div>")
    p.append("</details>")

    # -- Per-turn I/O --
    p.append('<div class="db-turns">')
    for i, utt in enumerate(state.transcript):
        is_ours = utt.role == env.role
        badge_cls = "db-trained" if is_ours else "db-opponent"
        badge_txt = "TRAINED" if is_ours else "OPPONENT"

        p.append(f'<div class="db-turn" data-role="{utt.role.value}">')

        # Turn header
        p.append('<div class="db-turn-hdr">')
        p.append(f'<span class="db-turn-num">Turn {i}</span>')
        p.append(f'<span class="db-role-tag" data-role="{utt.role.value}">{_esc(utt.role.value)}</span>')
        p.append(f'<span class="db-badge {badge_cls}">{badge_txt}</span>')
        p.append(f'<span class="db-phase">{_esc(utt.phase.value)} \u00b7 round {utt.round_index} \u00b7 {utt.token_count} tok</span>')
        p.append("</div>")

        # Input — show full assembled prompt (including instructions)
        truncated = replace(state, transcript=state.transcript[:i])
        assembled_msgs, _prefill = build_generation_messages(truncated, utt.role)
        input_text = _format_messages_text(tuple(assembled_msgs))
        p.append('<details class="db-io">')
        p.append(f"<summary>Input ({len(assembled_msgs)} messages)</summary>")
        p.append(f'<pre class="db-pre">{_esc(input_text)}</pre>')
        p.append("</details>")

        # Output (open by default)
        p.append('<details class="db-io" open>')
        p.append(f"<summary>Output ({utt.token_count} tokens)</summary>")
        p.append(f'<pre class="db-pre">{_esc(utt.text)}</pre>')
        p.append("</details>")

        p.append("</div>")  # /db-turn
    p.append("</div>")  # /db-turns

    # -- Judge verdict --
    if outcome and outcome.verdict_text:
        judge_msgs, _prefill = build_generation_messages(state, Role.JUDGE, trigger="final")
        judge_input = _format_messages_text(tuple(judge_msgs))

        p.append('<div class="db-judge">')
        p.append(f'<div class="db-judge-hdr">Judge Verdict: {_esc(winner_str)}</div>')

        p.append('<details class="db-io">')
        p.append(f"<summary>Judge Input ({len(judge_msgs)} messages)</summary>")
        p.append(f'<pre class="db-pre">{_esc(judge_input)}</pre>')
        p.append("</details>")

        p.append('<details class="db-io" open>')
        p.append("<summary>Judge Output (verdict)</summary>")
        p.append(f'<pre class="db-pre">{_esc(outcome.verdict_text)}</pre>')
        p.append("</details>")

        p.append("</div>")  # /db-judge

    p.append("</div>")  # /db-rollout
    return "\n".join(p)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
/* ---- debate trace ---- */

.db-batch {
    border-left: none !important;
    padding-left: 0 !important;
    background: white;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    padding: 1.25rem !important;
    margin: 1.25rem 0;
}
.db-batch > h2, .db-batch > h3 {
    font-size: 1.2rem;
    color: #1f2937;
    margin: 0 0 0.75rem 0;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 0.4rem;
}

.db-group {
    border-left: 4px solid #2563eb !important;
    padding-left: 1.25rem !important;
    margin: 0.75rem 0;
    background: #f8fafc;
    border-radius: 0 6px 6px 0;
    padding: 0.75rem 1rem 0.75rem 1.25rem !important;
}
.db-group > h3, .db-group > h4 {
    font-size: 0.95rem;
    color: #374151;
    margin: 0 0 0.5rem 0;
}
.db-group > .lt-section-body { padding-left: 0; }

/* --- rollout card --- */
.db-rollout {
    margin: 0.75rem 0;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    background: white;
    border: 1px solid #e5e7eb;
}
.db-rollout[data-result="win"]  { border-left: 4px solid #059669; }
.db-rollout[data-result="loss"] { border-left: 4px solid #dc2626; }
.db-rollout[data-result="tie"]  { border-left: 4px solid #6b7280; }

.db-summary {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    flex-wrap: wrap;
    margin-bottom: 0.6rem;
}

/* badges */
.db-badge {
    font-size: 0.7rem; font-weight: 600;
    padding: 0.1rem 0.45rem;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.db-role { color: #fff; }
.db-role[data-role="debater_a"] { background: #059669; }
.db-role[data-role="debater_b"] { background: #dc2626; }
.db-role[data-role="judge"]     { background: #2563eb; }
.db-trained  { background: #dbeafe; color: #1e40af; }
.db-opponent { background: #fef3c7; color: #92400e; }
.db-win  { background: #d1fae5; color: #065f46; }
.db-loss { background: #fee2e2; color: #991b1b; }
.db-tie  { background: #f3f4f6; color: #374151; }

.db-meta {
    font-size: 0.78rem;
    color: #6b7280;
    margin-left: auto;
}

/* system prompts */
.db-sysprompts {
    margin: 0.4rem 0;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
}
.db-sysprompts > summary {
    padding: 0.4rem 0.6rem;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.85rem;
    color: #4b5563;
}
.db-sysprompt {
    padding: 0.4rem 0.6rem;
    border-top: 1px solid #f3f4f6;
}
.db-sysprompt-hdr {
    font-size: 0.78rem; font-weight: 600;
    margin-bottom: 0.2rem;
}
.db-sysprompt-hdr[data-role="debater_a"] { color: #059669; }
.db-sysprompt-hdr[data-role="debater_b"] { color: #dc2626; }
.db-sysprompt-hdr[data-role="judge"]     { color: #2563eb; }

/* turns */
.db-turns { margin: 0.4rem 0; }
.db-turn {
    margin: 0.4rem 0;
    padding: 0.4rem 0.6rem;
    border-radius: 4px;
    border-left: 3px solid #e5e7eb;
}
.db-turn[data-role="debater_a"] { border-left-color: #059669; }
.db-turn[data-role="debater_b"] { border-left-color: #dc2626; }

.db-turn-hdr {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    flex-wrap: wrap;
    margin-bottom: 0.3rem;
}
.db-turn-num { font-weight: 700; font-size: 0.78rem; color: #374151; }
.db-role-tag { font-weight: 600; font-size: 0.78rem; }
.db-role-tag[data-role="debater_a"] { color: #059669; }
.db-role-tag[data-role="debater_b"] { color: #dc2626; }
.db-phase { font-size: 0.72rem; color: #6b7280; }

/* I/O blocks */
.db-io {
    margin: 0.2rem 0;
    border: 1px solid #f3f4f6;
    border-radius: 4px;
}
.db-io > summary {
    padding: 0.3rem 0.5rem;
    cursor: pointer;
    font-size: 0.78rem;
    font-weight: 500;
    color: #4b5563;
}
.db-pre {
    margin: 0;
    padding: 0.5rem;
    background: #f9fafb;
    font-family: "Courier New", monospace;
    font-size: 0.78rem;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 400px;
    overflow-y: auto;
}

/* judge */
.db-judge {
    margin: 0.6rem 0 0.2rem;
    padding: 0.6rem;
    border: 2px solid #2563eb;
    border-radius: 6px;
    background: #eff6ff;
}
.db-judge-hdr {
    font-weight: 700;
    font-size: 0.88rem;
    color: #1e40af;
    margin-bottom: 0.4rem;
}

/* cost report override */
.db-cost { border-left: none !important; padding-left: 0 !important; }
"""


class DebateTraceCSSInjector:
    """Formatter that only injects CSS (no body HTML)."""

    def to_html(self) -> str:
        return ""

    def get_css(self) -> str:
        return _CSS
