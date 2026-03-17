from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunData:
    log_dir: Path
    name: str  # log_dir.name (e.g. "rung1-no-think")
    config: dict | None  # from config.json
    metrics_rows: list[dict]  # from metrics.jsonl, ordered by step
    episode_rows: list[dict] | None  # from episodes/episodes.jsonl, or None


def _read_jsonl_safe(path: Path) -> list[dict]:
    """Read JSONL, skipping incomplete/corrupt lines."""
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_run(log_dir: Path) -> RunData:
    log_dir = Path(log_dir)

    # config.json
    config_path = log_dir / "config.json"
    config: dict | None = None
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            config = None

    # metrics.jsonl
    metrics_path = log_dir / "metrics.jsonl"
    metrics_rows = _read_jsonl_safe(metrics_path) if metrics_path.exists() else []

    # episodes/episodes.jsonl
    episodes_path = log_dir / "episodes" / "episodes.jsonl"
    episode_rows: list[dict] | None = None
    if episodes_path.exists():
        episode_rows = _read_jsonl_safe(episodes_path)

    return RunData(
        log_dir=log_dir,
        name=log_dir.name,
        config=config,
        metrics_rows=metrics_rows,
        episode_rows=episode_rows,
    )


def load_runs(dirs: list[Path]) -> list[RunData]:
    return [load_run(d) for d in dirs]
