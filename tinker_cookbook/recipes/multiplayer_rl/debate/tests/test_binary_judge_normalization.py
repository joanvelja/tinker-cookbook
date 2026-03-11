from __future__ import annotations

import os
import tempfile

from tinker_cookbook.recipes.multiplayer_rl.debate.prompts import resolve_prompts
from tinker_cookbook.scoring.types import normalize_binary_verdict_token


def _tmp_prompt_yaml() -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as handle:
        handle.write(
            """\
version: 2
system:
  judge:
    default: "judge"
  debater_a:
    default: "a"
  debater_b:
    default: "b"
question:
  debater_a: "q"
  debater_b: "q"
_matcher:
  system: "Compare two answers. Reply [YES] or (NO)."
  user: |
    Answer A: {{ a }}
    Answer B: {{ b }}
  positive: "[YES]"
  negative: "(NO)"
"""
        )
    return path


def test_loader_and_runtime_share_binary_verdict_normalization() -> None:
    path = _tmp_prompt_yaml()
    try:
        prompts = resolve_prompts(path)
        matcher = prompts.get_binary_judge_template("matcher")

        assert matcher is not None
        assert matcher.positive == normalize_binary_verdict_token("[YES]")
        assert matcher.negative == normalize_binary_verdict_token("(NO)")
    finally:
        resolve_prompts.cache_clear()
        os.unlink(path)
