"""CLI entry point: python -m scripts.spot_check <log_dir> [...]"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.text import Text

from .gates import GateResult, evaluate_gates
from .loaders import load_run
from .render import render_run
from .signals import compute_signals


def _resolve_dirs(raw: list[str]) -> list[Path]:
    """Expand globs and resolve to existing directories."""
    dirs: list[Path] = []
    for r in raw:
        p = Path(r)
        if "*" in r or "?" in r:
            parent = p.parent
            pattern = p.name
            matches = sorted(parent.glob(pattern))
            dirs.extend(m for m in matches if m.is_dir())
        elif p.is_dir():
            dirs.append(p)
        else:
            print(f"warning: {r} is not a directory, skipping", file=sys.stderr)
    return dirs


def _process_run(log_dir: Path, verbose: bool) -> Text | None:
    """Load, compute, evaluate, render one run. Returns None if no data."""
    run = load_run(log_dir)
    if not run.metrics_rows:
        print(f"warning: {log_dir} has no metrics.jsonl data, skipping", file=sys.stderr)
        return None
    signals = compute_signals(run)
    gates = evaluate_gates(signals)
    return render_run(run.name, gates, signals, verbose=verbose)


def _run_normal(dirs: list[Path], verbose: bool) -> None:
    console = Console(force_terminal=True)
    for d in dirs:
        output = _process_run(d, verbose)
        if output is not None:
            console.print(output)


def _snapshot_gates(dirs: list[Path]) -> dict[str, list[GateResult]]:
    """Return {run_name: [GateResult, ...]} for all dirs."""
    result: dict[str, list[GateResult]] = {}
    for d in dirs:
        run = load_run(d)
        if not run.metrics_rows:
            continue
        signals = compute_signals(run)
        result[run.name] = evaluate_gates(signals)
    return result


def _run_watch(dirs: list[Path], verbose: bool, interval: int) -> None:
    console = Console(force_terminal=True)
    event_log: list[str] = []
    prev_gates: dict[str, list[GateResult]] = {}
    prev_steps: dict[str, int] = {}

    try:
        with Live(Text("Loading..."), console=console, refresh_per_second=1) as live:
            while True:
                # Build full output
                output = Text()
                current_gates: dict[str, list[GateResult]] = {}

                for d in dirs:
                    run = load_run(d)
                    if not run.metrics_rows:
                        continue
                    signals = compute_signals(run)
                    gates = evaluate_gates(signals)
                    current_gates[run.name] = gates
                    step_count = len(run.metrics_rows)

                    # Detect gate status changes
                    old_step = prev_steps.get(run.name, 0)
                    if run.name in prev_gates and step_count != old_step:
                        old = {g.name: g.status for g in prev_gates[run.name]}
                        for g in gates:
                            old_status = old.get(g.name)
                            if old_status is not None and old_status != g.status:
                                event_log.append(
                                    f"  \u25b8 {run.name} step {old_step}\u2192{step_count}: "
                                    f"{g.name} {old_status.value}\u2192{g.status.value}"
                                )

                    prev_steps[run.name] = step_count
                    rendered = render_run(run.name, gates, signals, verbose=verbose)
                    output.append_text(rendered)

                prev_gates = current_gates

                # Event log footer
                if event_log:
                    output.append("\n")
                    output.append("Events:\n", style="bold dim")
                    for line in event_log[-10:]:
                        output.append(line + "\n", style="dim")

                live.update(output)
                time.sleep(interval)

    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="spot_check",
        description="Spot-check debate training runs",
    )
    parser.add_argument("log_dirs", nargs="+", help="Log directories (globs ok)")
    parser.add_argument("--watch", action="store_true", help="Live tailing mode")
    parser.add_argument("--verbose", action="store_true", help="Show secondary metrics")
    parser.add_argument("--interval", type=int, default=30, help="Watch poll interval (seconds)")
    args = parser.parse_args()

    dirs = _resolve_dirs(args.log_dirs)
    if not dirs:
        print("error: no valid log directories found", file=sys.stderr)
        sys.exit(1)

    if args.watch:
        _run_watch(dirs, args.verbose, args.interval)
    else:
        _run_normal(dirs, args.verbose)


if __name__ == "__main__":
    main()
