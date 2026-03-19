#!/usr/bin/env python3
"""Render RLVR rollout episodes as a self-contained HTML viewer.

Parses logtree trace HTML files from an RLVR training run and generates
a single-page viewer with per-episode cards, summary stats, and filtering.

Features:
  - Dark mode, clean typography, KaTeX for math rendering (requires internet)
  - Color-coded episodes: green (correct), red (incorrect), yellow (format issues)
  - Filtering by step, correct/incorrect, truncated/complete
  - Summary stats: total episodes, accuracy, format rate, truncation rate
  - Collapsible model responses with thinking block support

Usage:
    uv run python scripts/render_rlvr_html.py logs/gpqa_rl/smoke-run/
    uv run python scripts/render_rlvr_html.py logs/gpqa_rl/smoke-run/ -o viewer.html
    uv run python scripts/render_rlvr_html.py logs/gpqa_rl/smoke-run/ --step 5
    uv run python scripts/render_rlvr_html.py logs/gpqa_rl/smoke-run/ --filter correct
"""

import argparse
import html as html_mod
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    """One rollout episode parsed from logtree HTML."""

    step: int
    index_in_step: int  # episode index within the step
    question: str
    response: str
    reference: str
    # Grading signals — None means not parseable
    format_boxed: bool | None = None
    format_eos: bool | None = None
    correct: bool | None = None
    reward: float | None = None
    # Derived
    truncated: bool = False  # True if EOS was missing


@dataclass
class StepMetrics:
    """Aggregated metrics from metrics.jsonl for one step."""

    step: int
    correct: float = 0.0
    format_boxed: float = 0.0
    format_eos: float = 0.0
    format: float = 0.0
    reward_total: float = 0.0
    ac_tokens_per_turn: float = 0.0
    entropy: float = 0.0


# ---------------------------------------------------------------------------
# Parsing logtree HTML traces
# ---------------------------------------------------------------------------

# Match the four log_text lines from env.py / problem_env.py
_PROBLEM_RE = re.compile(r"^Problem:\s*(.+)", re.DOTALL)
_RESPONSE_RE = re.compile(r"^Response:\s*(.+)", re.DOTALL)
_REFERENCE_RE = re.compile(r"^Reference Answer:\s*(.+)", re.DOTALL)

# RLVREnv format: "Boxed: ✓/✗, EOS: ✓/✗, Correct: ✓/✗, Reward: 1.00"
_GRADE_RLVR_RE = re.compile(
    r"Boxed:\s*([✓✗]),\s*EOS:\s*([✓✗]),\s*Correct:\s*([✓✗]),\s*Reward:\s*([\-\d.]+)"
)
# ProblemEnv format: "Format Valid: ✓/✗, Correct: ✓/✗, Reward: 1.00"
_GRADE_PROBLEM_RE = re.compile(
    r"Format Valid:\s*([✓✗]),\s*Correct:\s*([✓✗]),\s*Reward:\s*([\-\d.]+)"
)


def _unescape(text: str) -> str:
    """Reverse HTML escaping."""
    return html_mod.unescape(text)


def _extract_lt_paragraphs(html_content: str) -> list[str]:
    """Extract text content from <p class="lt-p"> elements."""
    # Match <p class="lt-p">...</p> — the content is HTML-escaped text
    pattern = re.compile(r'<p\s+class="lt-p">\s*(.*?)\s*</p>', re.DOTALL)
    return [_unescape(m.group(1).strip()) for m in pattern.finditer(html_content)]


def _extract_step_from_title(html_content: str) -> int | None:
    """Extract step number from trace title like 'RL Iteration 5'."""
    m = re.search(r"<h1[^>]*>.*?RL Iteration\s+(\d+)", html_content, re.DOTALL)
    if m:
        return int(m.group(1))
    # Fallback: try filename pattern
    return None


def parse_trace_file(path: Path) -> list[Episode]:
    """Parse a logtree HTML trace file into a list of Episodes."""
    content = path.read_text(errors="replace")

    step = _extract_step_from_title(content)
    if step is None:
        # Try to extract from filename: train_iteration_000005.html
        m = re.search(r"iteration_(\d+)", path.stem)
        step = int(m.group(1)) if m else -1

    paragraphs = _extract_lt_paragraphs(content)

    episodes: list[Episode] = []
    # Walk through paragraphs looking for Problem/Response/Reference/Grade quads
    i = 0
    ep_idx = 0
    while i < len(paragraphs):
        p = paragraphs[i]
        m_prob = _PROBLEM_RE.match(p)
        if not m_prob:
            i += 1
            continue

        question = m_prob.group(1).strip()
        response = ""
        reference = ""
        format_boxed = None
        format_eos = None
        correct = None
        reward = None

        # Look ahead for Response, Reference, Grade (break if next Problem starts)
        for j in range(i + 1, min(i + 4, len(paragraphs))):
            pj = paragraphs[j]
            if _PROBLEM_RE.match(pj):
                break
            m_resp = _RESPONSE_RE.match(pj)
            if m_resp:
                response = m_resp.group(1).strip()
                continue
            m_ref = _REFERENCE_RE.match(pj)
            if m_ref:
                reference = m_ref.group(1).strip()
                continue
            m_grade = _GRADE_RLVR_RE.search(pj)
            if m_grade:
                format_boxed = m_grade.group(1) == "✓"
                format_eos = m_grade.group(2) == "✓"
                correct = m_grade.group(3) == "✓"
                reward = float(m_grade.group(4))
                continue
            m_grade2 = _GRADE_PROBLEM_RE.search(pj)
            if m_grade2:
                format_boxed = m_grade2.group(1) == "✓"
                format_eos = None  # ProblemEnv doesn't log EOS separately
                correct = m_grade2.group(2) == "✓"
                reward = float(m_grade2.group(3))
                continue

        episodes.append(
            Episode(
                step=step,
                index_in_step=ep_idx,
                question=question,
                response=response,
                reference=reference,
                format_boxed=format_boxed,
                format_eos=format_eos,
                correct=correct,
                reward=reward,
                truncated=format_eos is False if format_eos is not None else False,
            )
        )
        ep_idx += 1
        i += 1  # move past this Problem paragraph

    return episodes


def load_metrics(log_dir: Path) -> list[StepMetrics]:
    """Load metrics.jsonl if it exists."""
    path = log_dir / "metrics.jsonl"
    if not path.exists():
        return []

    metrics = []
    for line in path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue  # partial line from live training
        metrics.append(
            StepMetrics(
                step=d.get("step", -1),
                correct=d.get("env/all/correct", d.get("test/env/all/correct", 0.0)),
                format_boxed=d.get("env/all/format_boxed", d.get("env/all/format", 0.0)),
                format_eos=d.get("env/all/format_eos", 0.0),
                format=d.get("env/all/format", d.get("env/all/format_boxed", 0.0)),
                reward_total=d.get("env/all/reward/total", 0.0),
                ac_tokens_per_turn=d.get("env/all/ac_tokens_per_turn", 0.0),
                entropy=d.get("optim/entropy", 0.0),
            )
        )
    return metrics


def discover_traces(log_dir: Path) -> list[Path]:
    """Find all train_iteration_*.html files in the log dir."""
    traces = sorted(log_dir.glob("train_iteration_*.html"))
    if not traces:
        # Also check for eval traces
        traces = sorted(log_dir.glob("eval_*_iteration_*.html"))
    return traces


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    return html_mod.escape(text)


def _extract_think(text: str) -> tuple[str | None, str]:
    """Split <think>...</think> or <thinking>...</thinking> from response text."""
    m = re.search(r"<(think(?:ing)?)>(.*?)</\1>", text, re.DOTALL)
    if not m:
        return None, text
    thinking = m.group(2).strip()
    rest = (text[: m.start()] + text[m.end() :]).strip()
    return (thinking if thinking else None), rest


def _extract_boxed(text: str) -> str | None:
    """Extract \\boxed{...} content."""
    m = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_final_answer_tag(text: str) -> str | None:
    """Extract <final_answer>...</final_answer> content."""
    m = re.search(r"<final_answer>(.*?)</final_answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _verdict_class(ep: Episode) -> str:
    """CSS class for episode card border color."""
    if ep.correct:
        return "ep-correct"
    if ep.format_boxed is False or ep.truncated:
        return "ep-format-issue"
    return "ep-incorrect"


def _verdict_label(ep: Episode) -> str:
    if ep.correct:
        return "CORRECT"
    if ep.truncated:
        return "TRUNCATED"
    if ep.format_boxed is False:
        return "FORMAT ERROR"
    return "INCORRECT"


def _render_episode_html(ep: Episode, global_idx: int) -> str:
    """Render one episode card."""
    vc = _verdict_class(ep)
    vl = _verdict_label(ep)
    reward_str = f"{ep.reward:+.2f}" if ep.reward is not None else "?"

    thinking, visible_response = _extract_think(ep.response)
    extracted = _extract_final_answer_tag(ep.response) or _extract_boxed(visible_response)

    # Build grading pills
    pills = []
    if ep.format_boxed is not None:
        cls = "pill-ok" if ep.format_boxed else "pill-bad"
        pills.append(f'<span class="pill {cls}">format {"✓" if ep.format_boxed else "✗"}</span>')
    if ep.format_eos is not None:
        cls = "pill-ok" if ep.format_eos else "pill-bad"
        pills.append(f'<span class="pill {cls}">eos {"✓" if ep.format_eos else "✗"}</span>')
    if ep.correct is not None:
        cls = "pill-ok" if ep.correct else "pill-bad"
        pills.append(f'<span class="pill {cls}">correct {"✓" if ep.correct else "✗"}</span>')
    pills.append(f'<span class="pill pill-reward">reward {_esc(reward_str)}</span>')

    pills_html = " ".join(pills)

    # Question (truncated in header, full in expandable)
    q_preview = ep.question[:120].replace("\n", " ")
    if len(ep.question) > 120:
        q_preview += "..."

    parts = [
        f'<div class="ep-card {vc}" '
        f'data-step="{ep.step}" '
        f'data-correct="{1 if ep.correct else 0}" '
        f'data-truncated="{1 if ep.truncated else 0}" '
        f'data-format="{1 if ep.format_boxed else 0}" '
        f'id="ep-{global_idx}">',
        f'<div class="ep-hdr" onclick="toggleCard(this)">',
        f'  <span class="ep-idx">#{global_idx}</span>',
        f'  <span class="ep-step">step {ep.step}</span>',
        f'  <span class="ep-verdict {vc}">{_esc(vl)}</span>',
        f'  <span class="ep-q-preview">{_esc(q_preview)}</span>',
        f'  <span class="ep-expand-icon">▶</span>',
        f"</div>",
        f'<div class="ep-body" style="display:none">',
        # Grading pills
        f'  <div class="ep-pills">{pills_html}</div>',
        # Question
        f'  <div class="ep-section">',
        f'    <div class="ep-label">Question</div>',
        f'    <div class="ep-content output-rendered">{_esc(ep.question)}</div>',
        f"  </div>",
    ]

    # Thinking (if present)
    if thinking:
        parts.extend([
            f'  <details class="think-block">',
            f'    <summary class="think-summary">reasoning ({len(thinking)} chars)</summary>',
            f'    <pre class="think-content">{_esc(thinking)}</pre>',
            f"  </details>",
        ])

    # Response
    parts.extend([
        f'  <div class="ep-section">',
        f'    <div class="ep-label">Response</div>',
        f'    <div class="ep-content output-rendered">{_esc(visible_response)}</div>',
        f"  </div>",
    ])

    # Extracted answer + Reference
    if extracted:
        parts.extend([
            f'  <div class="ep-section ep-section-inline">',
            f'    <div class="ep-label">Extracted</div>',
            f'    <div class="ep-content ep-answer"><code>{_esc(extracted)}</code></div>',
            f"  </div>",
        ])

    parts.extend([
        f'  <div class="ep-section ep-section-inline">',
        f'    <div class="ep-label">Reference</div>',
        f'    <div class="ep-content ep-answer"><code>{_esc(ep.reference)}</code></div>',
        f"  </div>",
    ])

    parts.extend([
        f"</div>",  # ep-body
        f"</div>",  # ep-card
    ])

    return "\n".join(parts)


def _render_summary_stats(episodes: list[Episode], metrics: list[StepMetrics]) -> str:
    """Render summary statistics bar."""
    total = len(episodes)
    if total == 0:
        return '<div class="summary">No episodes found.</div>'

    n_correct = sum(1 for e in episodes if e.correct)
    n_format = sum(1 for e in episodes if e.format_boxed)
    n_truncated = sum(1 for e in episodes if e.truncated)
    steps = sorted(set(e.step for e in episodes))

    accuracy = n_correct / total * 100
    format_rate = n_format / total * 100
    trunc_rate = n_truncated / total * 100

    parts = [
        '<div class="summary">',
        f'  <span class="stat"><b>{total}</b> episodes</span>',
        f'  <span class="stat"><b>{len(steps)}</b> steps ({min(steps)}–{max(steps)})</span>',
        f'  <span class="stat stat-correct"><b>{accuracy:.1f}%</b> accuracy ({n_correct}/{total})</span>',
        f'  <span class="stat stat-format"><b>{format_rate:.1f}%</b> format ({n_format}/{total})</span>',
        f'  <span class="stat stat-trunc"><b>{trunc_rate:.1f}%</b> truncated ({n_truncated}/{total})</span>',
        "</div>",
    ]
    return "\n".join(parts)


def _render_filters_html(episodes: list[Episode]) -> str:
    """Render filter controls."""
    steps = sorted(set(e.step for e in episodes))
    step_options = "".join(f'<option value="{s}">Step {s}</option>' for s in steps)

    return f"""<div class="filters">
  <select id="filter-step" onchange="applyFilters()">
    <option value="all">All steps</option>
    {step_options}
  </select>
  <select id="filter-verdict" onchange="applyFilters()">
    <option value="all">All verdicts</option>
    <option value="correct">Correct only</option>
    <option value="incorrect">Incorrect only</option>
    <option value="truncated">Truncated only</option>
  </select>
  <span class="filter-count" id="filter-count"></span>
</div>"""


# ---------------------------------------------------------------------------
# CSS & JS
# ---------------------------------------------------------------------------

_CSS = """
<style>
:root {
  --bg: #1a1b26;
  --bg-card: #24283b;
  --bg-card-hover: #292e42;
  --text: #c0caf5;
  --text-dim: #565f89;
  --text-bright: #e0e6ff;
  --border: #3b4261;
  --green: #9ece6a;
  --green-bg: rgba(158, 206, 106, 0.08);
  --red: #f7768e;
  --red-bg: rgba(247, 118, 142, 0.08);
  --yellow: #e0af68;
  --yellow-bg: rgba(224, 175, 104, 0.08);
  --blue: #7aa2f7;
  --purple: #bb9af7;
  --mono: "Berkeley Mono", "JetBrains Mono", "Fira Code", "Cascadia Code", monospace;
  --sans: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: var(--sans);
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  padding: 1.5rem;
  max-width: 1100px;
  margin: 0 auto;
}

h1 {
  font-size: 1.3rem;
  font-weight: 600;
  color: var(--text-bright);
  margin-bottom: 0.25rem;
}

.subtitle {
  color: var(--text-dim);
  font-size: 0.8rem;
  margin-bottom: 1rem;
}

/* Summary bar */
.summary {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  padding: 0.75rem 1rem;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 0.75rem;
  font-size: 0.82rem;
}

.stat { color: var(--text-dim); }
.stat b { color: var(--text-bright); }
.stat-correct b { color: var(--green); }
.stat-format b { color: var(--blue); }
.stat-trunc b { color: var(--yellow); }

/* Filters */
.filters {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  margin-bottom: 1rem;
  font-size: 0.82rem;
}

.filters select {
  background: var(--bg-card);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.4rem 0.6rem;
  font-family: var(--sans);
  font-size: 0.82rem;
  cursor: pointer;
}

.filter-count {
  color: var(--text-dim);
  font-size: 0.78rem;
}

/* Episode cards */
.ep-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 0.5rem;
  border-left: 3px solid var(--border);
  transition: border-color 0.15s;
}

.ep-card.ep-correct { border-left-color: var(--green); }
.ep-card.ep-incorrect { border-left-color: var(--red); }
.ep-card.ep-format-issue { border-left-color: var(--yellow); }

.ep-hdr {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.5rem 0.75rem;
  cursor: pointer;
  user-select: none;
  font-size: 0.82rem;
}

.ep-hdr:hover { background: var(--bg-card-hover); border-radius: 8px; }

.ep-idx {
  color: var(--text-dim);
  font-family: var(--mono);
  font-size: 0.75rem;
  min-width: 2.5rem;
}

.ep-step {
  color: var(--purple);
  font-size: 0.75rem;
  font-family: var(--mono);
}

.ep-verdict {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: 0.15rem 0.4rem;
  border-radius: 4px;
}

.ep-verdict.ep-correct { color: var(--green); background: var(--green-bg); }
.ep-verdict.ep-incorrect { color: var(--red); background: var(--red-bg); }
.ep-verdict.ep-format-issue { color: var(--yellow); background: var(--yellow-bg); }

.ep-q-preview {
  color: var(--text-dim);
  font-size: 0.78rem;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ep-expand-icon {
  color: var(--text-dim);
  font-size: 0.65rem;
  transition: transform 0.2s;
}

.ep-card.expanded .ep-expand-icon { transform: rotate(90deg); }

/* Episode body */
.ep-body {
  padding: 0.5rem 0.75rem 0.75rem;
  border-top: 1px solid var(--border);
}

.ep-pills {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-bottom: 0.75rem;
}

.pill {
  font-size: 0.72rem;
  font-family: var(--mono);
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  font-weight: 500;
}

.pill-ok { color: var(--green); background: var(--green-bg); }
.pill-bad { color: var(--red); background: var(--red-bg); }
.pill-reward { color: var(--blue); background: rgba(122, 162, 247, 0.1); }

.ep-section {
  margin-bottom: 0.75rem;
}

.ep-section-inline {
  display: inline-block;
  margin-right: 1.5rem;
}

.ep-label {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-dim);
  margin-bottom: 0.25rem;
}

.ep-content {
  font-size: 0.82rem;
  line-height: 1.6;
  white-space: pre-wrap;
  word-wrap: break-word;
  max-height: 500px;
  overflow-y: auto;
  padding: 0.5rem;
  background: rgba(0, 0, 0, 0.15);
  border-radius: 6px;
  border: 1px solid var(--border);
}

.ep-answer {
  font-family: var(--mono);
  font-weight: 600;
  color: var(--text-bright);
  background: rgba(122, 162, 247, 0.08);
  border-color: var(--blue);
  max-height: none;
  display: inline-block;
}

/* Thinking blocks */
.think-block {
  margin-bottom: 0.75rem;
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
}

.think-summary {
  padding: 0.4rem 0.6rem;
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--purple);
  cursor: pointer;
  user-select: none;
  background: rgba(187, 154, 247, 0.05);
}

.think-content {
  padding: 0.5rem 0.6rem;
  font-family: var(--mono);
  font-size: 0.78rem;
  white-space: pre-wrap;
  word-wrap: break-word;
  max-height: 400px;
  overflow-y: auto;
  background: rgba(0, 0, 0, 0.1);
  color: var(--text-dim);
}

/* KaTeX overrides for dark mode */
.katex { color: var(--text-bright); }
.katex .mord, .katex .mbin, .katex .mrel,
.katex .mopen, .katex .mclose, .katex .mpunct { color: var(--text-bright); }
</style>
"""

_JS = """
<script>
function toggleCard(hdr) {
  const card = hdr.parentElement;
  const body = card.querySelector('.ep-body');
  if (body.style.display === 'none') {
    body.style.display = 'block';
    card.classList.add('expanded');
  } else {
    body.style.display = 'none';
    card.classList.remove('expanded');
  }
}

function applyFilters() {
  const stepVal = document.getElementById('filter-step').value;
  const verdictVal = document.getElementById('filter-verdict').value;
  const cards = document.querySelectorAll('.ep-card');
  let shown = 0;
  let total = cards.length;

  cards.forEach(card => {
    let show = true;

    if (stepVal !== 'all' && card.dataset.step !== stepVal) {
      show = false;
    }

    if (verdictVal === 'correct' && card.dataset.correct !== '1') show = false;
    if (verdictVal === 'incorrect' && card.dataset.correct !== '0') show = false;
    if (verdictVal === 'truncated' && card.dataset.truncated !== '1') show = false;

    card.style.display = show ? 'block' : 'none';
    if (show) shown++;
  });

  document.getElementById('filter-count').textContent =
    shown === total ? '' : `showing ${shown} of ${total}`;
}

function expandAll() {
  document.querySelectorAll('.ep-card').forEach(card => {
    if (card.style.display === 'none') return;
    const body = card.querySelector('.ep-body');
    body.style.display = 'block';
    card.classList.add('expanded');
  });
}

function collapseAll() {
  document.querySelectorAll('.ep-card').forEach(card => {
    const body = card.querySelector('.ep-body');
    body.style.display = 'none';
    card.classList.remove('expanded');
  });
}

// KaTeX auto-render on load
document.addEventListener('DOMContentLoaded', () => {
  if (typeof renderMathInElement === 'function') {
    document.querySelectorAll('.output-rendered, .ep-answer').forEach(el => {
      renderMathInElement(el, {
        delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '$', right: '$', display: false},
          {left: '\\\\(', right: '\\\\)', display: false},
          {left: '\\\\[', right: '\\\\]', display: true},
        ],
        throwOnError: false,
      });
    });
  }
});
</script>
"""


def render_viewer_html(
    episodes: list[Episode],
    metrics: list[StepMetrics],
    log_dir: str,
) -> str:
    """Generate the full self-contained HTML viewer."""
    summary = _render_summary_stats(episodes, metrics)
    filters = _render_filters_html(episodes)
    episode_cards = "\n".join(
        _render_episode_html(ep, i) for i, ep in enumerate(episodes)
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RLVR Rollouts — {_esc(log_dir)}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/contrib/auto-render.min.js"></script>
{_CSS}
</head>
<body>

<h1>RLVR Rollout Viewer</h1>
<div class="subtitle">{_esc(log_dir)}</div>

{summary}
{filters}

<div style="margin-bottom:0.75rem; display:flex; gap:0.5rem;">
  <button onclick="expandAll()" style="background:var(--bg-card);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:0.3rem 0.6rem;cursor:pointer;font-size:0.78rem;">Expand all</button>
  <button onclick="collapseAll()" style="background:var(--bg-card);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:0.3rem 0.6rem;cursor:pointer;font-size:0.78rem;">Collapse all</button>
</div>

<div id="episodes">
{episode_cards}
</div>

{_JS}
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Render RLVR rollout episodes as an HTML viewer."
    )
    parser.add_argument("log_dir", help="Path to RLVR log directory")
    parser.add_argument("-o", "--output", default=None, help="Output HTML path")
    parser.add_argument("--step", type=int, default=None, help="Filter to specific step")
    parser.add_argument(
        "--filter",
        choices=["correct", "incorrect", "truncated"],
        default=None,
        help="Filter episodes by verdict",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"Error: {log_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Discover and parse traces
    traces = discover_traces(log_dir)
    if not traces:
        print(f"No trace files found in {log_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(traces)} trace files", file=sys.stderr)

    all_episodes: list[Episode] = []
    for trace_path in traces:
        eps = parse_trace_file(trace_path)
        all_episodes.extend(eps)
        if eps:
            print(
                f"  {trace_path.name}: {len(eps)} episodes (step {eps[0].step})",
                file=sys.stderr,
            )

    if not all_episodes:
        print("No episodes parsed from trace files", file=sys.stderr)
        sys.exit(1)

    # Apply CLI filters
    if args.step is not None:
        all_episodes = [e for e in all_episodes if e.step == args.step]
    if args.filter == "correct":
        all_episodes = [e for e in all_episodes if e.correct]
    elif args.filter == "incorrect":
        all_episodes = [e for e in all_episodes if not e.correct]
    elif args.filter == "truncated":
        all_episodes = [e for e in all_episodes if e.truncated]

    # Load metrics
    metrics = load_metrics(log_dir)

    # Render
    html = render_viewer_html(all_episodes, metrics, str(log_dir))

    # Output
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = log_dir / "rollout_viewer.html"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"\n{len(all_episodes)} episodes → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
