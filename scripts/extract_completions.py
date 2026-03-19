#!/usr/bin/env python3
"""Extract structured completion records from logtree HTML files.

Parses HTML files produced by logtree during RL training, extracting
Problem/Response/Reference Answer/reward info logged by problem_env.py.

Output: JSONL to stdout.
"""

import argparse
import html
import json
import os
import sys
from html.parser import HTMLParser


class LogtreeExtractor(HTMLParser):
    """Extract completion records from a single logtree HTML file.

    The structure is:
        <section class="lt-section">  (do_single_rollout)
            <p class="lt-p">Problem: ...</p>
            <p class="lt-p">Response: ...</p>
            <p class="lt-p">Reference Answer: ...</p>
            <p class="lt-p">Format Valid: ..., Correct: ..., Reward: ...</p>
        </section>
    """

    def __init__(self):
        super().__init__()
        self.records: list[dict] = []

        # State tracking
        self._in_lt_p = False
        self._current_text = ""
        self._paragraphs: list[str] = []  # collected <p class="lt-p"> texts in current rollout section
        self._in_rollout_section = False
        self._section_depth = 0  # nesting depth within rollout section

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        attrs_dict = dict(attrs)

        if tag == "p" and attrs_dict.get("class") == "lt-p":
            self._in_lt_p = True
            self._current_text = ""
            return

        if tag == "section" and self._in_rollout_section:
            self._section_depth += 1

        # Detect entering a do_single_rollout section:
        # It's an h3 with text "do_single_rollout" inside a section.
        # We track sections and collect paragraphs within them.
        if tag == "h3":
            self._pending_h3 = True
            self._h3_text = ""

    def handle_endtag(self, tag: str):
        if tag == "p" and self._in_lt_p:
            self._in_lt_p = False
            text = self._current_text.strip()
            if self._in_rollout_section and text:
                self._paragraphs.append(text)
            self._current_text = ""
            return

        if tag == "h3" and hasattr(self, "_pending_h3") and self._pending_h3:
            self._pending_h3 = False
            if "do_single_rollout" in self._h3_text.strip():
                # Entering a new rollout section
                if self._in_rollout_section:
                    self._flush_rollout()
                self._in_rollout_section = True
                self._section_depth = 0
                self._paragraphs = []
            self._h3_text = ""

        if tag == "section" and self._in_rollout_section:
            if self._section_depth > 0:
                self._section_depth -= 1
            else:
                # Exiting the rollout section
                self._flush_rollout()
                self._in_rollout_section = False

    def handle_data(self, data: str):
        if self._in_lt_p:
            self._current_text += data
        if hasattr(self, "_pending_h3") and self._pending_h3:
            self._h3_text += data

    def handle_entityref(self, name: str):
        char = html.unescape(f"&{name};")
        if self._in_lt_p:
            self._current_text += char
        if hasattr(self, "_pending_h3") and self._pending_h3:
            self._h3_text += char

    def handle_charref(self, name: str):
        char = html.unescape(f"&#{name};")
        if self._in_lt_p:
            self._current_text += char
        if hasattr(self, "_pending_h3") and self._pending_h3:
            self._h3_text += char

    def _flush_rollout(self):
        """Parse collected paragraphs into a record."""
        if not self._paragraphs:
            return

        problem = None
        response = None
        reference_answer = None
        format_valid = None
        correct = None
        reward = None

        # The paragraphs follow a known order:
        #   1. "Problem: ..."
        #   2. "Response: ..."
        #   3. "Reference Answer: ..."
        #   4. "Format Valid: ..., Correct: ..., Reward: ..."
        #
        # But the Response can be very long and is always a single <p>.
        # We match by prefix.
        for p in self._paragraphs:
            if p.startswith("Problem:"):
                problem = p[len("Problem:"):].strip()
            elif p.startswith("Response:"):
                response = p[len("Response:"):].strip()
            elif p.startswith("Reference Answer:"):
                reference_answer = p[len("Reference Answer:"):].strip()
            elif p.startswith("Format Valid:"):
                # Parse: "Format Valid: ✓, Correct: ✗, Reward: -0.10"
                rest = p[len("Format Valid:"):].strip()
                parts = [x.strip() for x in rest.split(",")]
                for part in parts:
                    if part.startswith("Correct:"):
                        correct = part.split(":")[1].strip() == "✓"
                    elif part.startswith("Reward:"):
                        try:
                            reward = float(part.split(":")[1].strip())
                        except ValueError:
                            reward = None
                    else:
                        # First part is the format_valid value itself
                        format_valid = part.strip() == "✓"

        if problem is not None or response is not None:
            self.records.append({
                "problem": problem,
                "response": response,
                "reference_answer": reference_answer,
                "format_valid": format_valid,
                "correct": correct,
                "reward": reward,
            })

        self._paragraphs = []

    def close(self):
        # Flush any remaining rollout
        if self._in_rollout_section:
            self._flush_rollout()
        super().close()


def extract_from_file(filepath: str) -> list[dict]:
    """Extract all completion records from a single HTML file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    parser = LogtreeExtractor()
    parser.feed(content)
    parser.close()

    filename = os.path.basename(filepath)
    for rec in parser.records:
        rec["file"] = filename

    return parser.records


def main():
    parser = argparse.ArgumentParser(
        description="Extract completions from logtree HTML files as JSONL."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="logs/gpqa_rl/smoke-nothink-100b-1773198629/",
        help="Directory containing logtree HTML files",
    )
    args = parser.parse_args()

    directory = args.directory
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    html_files = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".html")
    )

    if not html_files:
        print(f"No HTML files found in {directory}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(html_files)} HTML files...", file=sys.stderr)

    total = 0
    for filepath in html_files:
        records = extract_from_file(filepath)
        for rec in records:
            print(json.dumps(rec, ensure_ascii=False))
            total += 1

    print(f"Extracted {total} records.", file=sys.stderr)


if __name__ == "__main__":
    main()
