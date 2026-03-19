"""Tests for RLVR spot-check gates using synthetic metrics.jsonl data."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts.spot_check_rlvr.gates import GateStatus, evaluate_gates
from scripts.spot_check_rlvr.loaders import load_run
from scripts.spot_check_rlvr.signals import compute_signals


def _write_metrics(tmp: Path, rows: list[dict]) -> Path:
    metrics_path = tmp / "metrics.jsonl"
    with open(metrics_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return tmp


def _gate_by_name(gates, name: str):
    for g in gates:
        if g.name == name:
            return g
    raise KeyError(name)


class TestLearningGate:
    def test_improving_correct_rate_is_ok(self, tmp_path):
        rows = [
            {"step": i, "env/all/correct": 0.1 + i * 0.02, "env/all/reward/total": 0.1 + i * 0.02}
            for i in range(15)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        assert _gate_by_name(gates, "LEARNING").status == GateStatus.OK

    def test_declining_correct_rate_fails(self, tmp_path):
        rows = [
            {"step": i, "env/all/correct": 0.5 - i * 0.02, "env/all/reward/total": 0.5 - i * 0.02}
            for i in range(15)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        assert _gate_by_name(gates, "LEARNING").status == GateStatus.FAIL

    def test_flat_correct_rate_warns(self, tmp_path):
        rows = [
            {"step": i, "env/all/correct": 0.3, "env/all/reward/total": 0.3}
            for i in range(15)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        assert _gate_by_name(gates, "LEARNING").status == GateStatus.WARN

    def test_reward_flatline_fails(self, tmp_path):
        rows = [
            {"step": i, "env/all/correct": 0.3 + i * 0.01, "env/all/reward/total": 0.0}
            for i in range(5)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        assert _gate_by_name(gates, "LEARNING").status == GateStatus.FAIL

    def test_all_correct_flat_is_ok(self, tmp_path):
        """100% accuracy with flat slope should be OK, not WARN."""
        rows = [
            {"step": i, "env/all/correct": 1.0, "env/all/reward/total": 1.0}
            for i in range(15)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        assert _gate_by_name(gates, "LEARNING").status == GateStatus.OK


class TestFormatGate:
    def test_high_format_rates_ok(self, tmp_path):
        rows = [
            {"step": i, "env/all/format_boxed": 0.95, "env/all/format_eos": 0.95}
            for i in range(5)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        assert _gate_by_name(gates, "FORMAT").status == GateStatus.OK

    def test_low_format_boxed_warns(self, tmp_path):
        rows = [
            {"step": i, "env/all/format_boxed": 0.7, "env/all/format_eos": 0.95}
            for i in range(5)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        assert _gate_by_name(gates, "FORMAT").status == GateStatus.WARN

    def test_very_low_format_fails(self, tmp_path):
        rows = [
            {"step": i, "env/all/format_boxed": 0.3, "env/all/format_eos": 0.3}
            for i in range(5)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        assert _gate_by_name(gates, "FORMAT").status == GateStatus.FAIL

    def test_legacy_format_key_works(self, tmp_path):
        rows = [
            {"step": i, "env/all/format": 0.4}
            for i in range(5)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        assert _gate_by_name(gates, "FORMAT").status == GateStatus.FAIL

    def test_extract_fail_rate_computed(self, tmp_path):
        """extract_fail_rate = 1 - format_boxed, lives in FORMAT gate signals."""
        rows = [
            {"step": i, "env/all/format_boxed": 0.8}
            for i in range(5)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        efr = signals.series.get("extract_fail_rate", [])
        assert len(efr) == 5
        assert all(abs(v - 0.2) < 1e-9 for v in efr)


class TestCompressionGate:
    def test_short_responses_ok(self, tmp_path):
        rows = [
            {"step": i, "env/all/ac_tokens_per_turn": 500}
            for i in range(5)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        assert _gate_by_name(gates, "COMPRESSION").status == GateStatus.OK

    def test_long_responses_warn(self, tmp_path):
        rows = [
            {"step": i, "env/all/ac_tokens_per_turn": 3500}
            for i in range(5)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        assert _gate_by_name(gates, "COMPRESSION").status == GateStatus.WARN

    def test_very_long_responses_fail(self, tmp_path):
        rows = [
            {"step": i, "env/all/ac_tokens_per_turn": 5000}
            for i in range(5)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        assert _gate_by_name(gates, "COMPRESSION").status == GateStatus.FAIL

    def test_rapidly_growing_responses_fail(self, tmp_path):
        rows = [
            {"step": i, "env/all/ac_tokens_per_turn": 500 + i * 100}
            for i in range(15)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        assert _gate_by_name(gates, "COMPRESSION").status == GateStatus.FAIL


class TestGraderGate:
    def test_fast_grading_ok(self, tmp_path):
        rows = [
            {"step": i, "env/all/time/check_answer_s": 0.5}
            for i in range(5)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        assert _gate_by_name(gates, "GRADER").status == GateStatus.OK

    def test_slow_grading_warns(self, tmp_path):
        rows = [
            {"step": i, "env/all/time/check_answer_s": 3.0}
            for i in range(5)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        assert _gate_by_name(gates, "GRADER").status == GateStatus.WARN

    def test_very_slow_grading_fails(self, tmp_path):
        rows = [
            {"step": i, "env/all/time/check_answer_s": 6.0}
            for i in range(5)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        assert _gate_by_name(gates, "GRADER").status == GateStatus.FAIL

    def test_no_latency_data_ok(self, tmp_path):
        """No check_answer_s in metrics => GRADER gate is OK (no data = no alarm)."""
        rows = [
            {"step": i, "env/all/correct": 0.5}
            for i in range(5)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        assert _gate_by_name(gates, "GRADER").status == GateStatus.OK


class TestOverfitGate:
    def test_no_eval_data_ok(self, tmp_path):
        rows = [
            {"step": i, "env/all/correct": 0.5}
            for i in range(5)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        assert _gate_by_name(gates, "OVERFIT").status == GateStatus.OK

    def test_small_gap_ok(self, tmp_path):
        rows = [
            {
                "step": i,
                "env/all/correct": 0.5 + i * 0.01,
                "test/env/all/correct": 0.48 + i * 0.01,
            }
            for i in range(10)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        assert _gate_by_name(gates, "OVERFIT").status == GateStatus.OK

    def test_no_forward_fill(self, tmp_path):
        """train_eval_gap should be None at steps without eval data."""
        rows = [
            {"step": i, "env/all/correct": 0.5 + i * 0.02}
            for i in range(10)
        ]
        rows[0]["test/env/all/correct"] = 0.50
        rows[5]["test/env/all/correct"] = 0.55
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gap = signals.series["train_eval_gap"]
        # Only steps 0 and 5 should have values
        assert gap[0] is not None  # 0.5 - 0.5 = 0.0
        assert gap[5] is not None  # 0.6 - 0.55 = 0.05
        assert gap[1] is None  # no eval at step 1
        assert gap[9] is None  # no eval at step 9

    def test_large_gap_fails(self, tmp_path):
        # Train improving, eval declining => overfitting
        # Eval at steps 0 and 19 only. At step 19: train=0.87, eval=0.35 => gap=0.52
        rows = [
            {
                "step": i,
                "env/all/correct": 0.3 + i * 0.03,
            }
            for i in range(20)
        ]
        # Eval at start and end: declining while train improves
        rows[0]["test/env/all/correct"] = 0.50
        rows[19]["test/env/all/correct"] = 0.35
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        assert _gate_by_name(gates, "OVERFIT").status == GateStatus.FAIL


class TestEndToEnd:
    def test_healthy_run_all_ok(self, tmp_path):
        """A healthy RLVR run: improving correct, good format, moderate tokens."""
        rows = [
            {
                "step": i,
                "env/all/correct": 0.2 + i * 0.02,
                "env/all/reward/total": 0.2 + i * 0.02,
                "env/all/format_boxed": 0.9 + i * 0.005,
                "env/all/format_eos": 0.95,
                "env/all/ac_tokens_per_turn": 800 + i * 5,
                "env/all/time/check_answer_s": 0.3,
                "optim/entropy": 0.4,
                "optim/kl_sample_train_v2": 0.002,
                "test/env/all/correct": 0.2 + i * 0.018 if i % 5 == 0 else None,
            }
            for i in range(20)
        ]
        # Ensure eval data at endpoints
        rows[0]["test/env/all/correct"] = 0.20
        rows[19]["test/env/all/correct"] = 0.54
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)

        for g in gates:
            assert g.status == GateStatus.OK, f"{g.name} should be OK, got {g.status}"

    def test_all_signals_computed(self, tmp_path):
        """All signal series should have correct length."""
        rows = [
            {
                "step": i,
                "env/all/correct": 0.3,
                "env/all/reward/total": 0.3,
                "env/all/format_boxed": 0.9,
                "env/all/format_eos": 0.9,
                "env/all/ac_tokens_per_turn": 800,
            }
            for i in range(10)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)

        for name, series in signals.series.items():
            assert len(series) == 10, f"Signal {name} has {len(series)} values, expected 10"

    def test_streak_calculation(self, tmp_path):
        """Streak should count consecutive steps at same gate status."""
        rows = [
            {"step": i, "env/all/correct": 0.3, "env/all/reward/total": 0.3}
            for i in range(10)
        ]
        _write_metrics(tmp_path, rows)
        run = load_run(tmp_path)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        learning = _gate_by_name(gates, "LEARNING")
        assert learning.streak >= 2

    def test_empty_run_no_crash(self, tmp_path):
        """Empty metrics.jsonl shouldn't crash."""
        (tmp_path / "metrics.jsonl").write_text("")
        run = load_run(tmp_path)
        assert run.metrics_rows == []

    def test_real_gpqa_data(self):
        """Run against actual GPQA metrics if available."""
        data_dir = Path(__file__).resolve().parents[3] / "logs" / "gpqa_rl" / "smoke-nothink-100b-1773198629"
        if not data_dir.exists():
            return  # skip if no data
        run = load_run(data_dir)
        signals = compute_signals(run)
        gates = evaluate_gates(signals)
        # Just check it doesn't crash and produces results
        assert len(gates) == 5
        for g in gates:
            assert g.status in (GateStatus.OK, GateStatus.WARN, GateStatus.FAIL)
