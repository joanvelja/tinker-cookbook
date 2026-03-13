"""Tests for JSONL episode sidecar logging (on_group_complete override)."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from unittest.mock import MagicMock

import pytest

from tinker_cookbook.recipes.multiplayer_rl.debate.env import DebateEnv
from tinker_cookbook.recipes.multiplayer_rl.debate.builders import (
    DebateGroupBuilder,
    _EPISODE_LOG_LOCK,
)
from tinker_cookbook.recipes.multiplayer_rl.debate.core.runtime import DebateRuntime
from tinker_cookbook.recipes.multiplayer_rl.debate.core.schedule import build_schedule
from tinker_cookbook.recipes.multiplayer_rl.debate.types import (
    DebateGameSpec,
    DebateOutcome,
    DebateProblemSpec,
    DebateSpec,
    DebateState,
    Phase,
    ProtocolKind,
    Role,
    ScoringMode,
    Utterance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    *,
    trained_role: Role = Role.DEBATER_A,
    target: str | None = "B",
    transcript: tuple[Utterance, ...] = (),
    outcome: DebateOutcome | None = None,
) -> DebateState:
    schedule = build_schedule(ProtocolKind.SEQUENTIAL, 1)
    spec = DebateSpec(
        debate_id=uuid.uuid4().hex,
        problem=DebateProblemSpec(
            task_prompt="What is 2+2?",
            scoring_mode=ScoringMode.MCQ,
            answer_by_role={Role.DEBATER_A: "A", Role.DEBATER_B: "B"},
            target=target,
        ),
        schedule=schedule,
        protocol_kind=ProtocolKind.SEQUENTIAL,
        prompts_ref="default",
    )
    return DebateState(
        spec=spec,
        slot_index=len(schedule),
        rounds_completed=1,
        transcript=transcript,
        pending_simultaneous={},
        judge_trace=(),
        done=True,
        outcome=outcome,
    )


def _make_utterances(trained_role: Role = Role.DEBATER_A) -> tuple[Utterance, ...]:
    opponent_role = Role.DEBATER_B if trained_role == Role.DEBATER_A else Role.DEBATER_A
    return (
        Utterance(
            role=trained_role,
            round_index=0,
            phase=Phase.PROPOSE,
            text="I argue for A.\n\nAnswer: A",
            token_count=10,
            slot_id=0,
            fields={"answer": "A"},
        ),
        Utterance(
            role=opponent_role,
            round_index=0,
            phase=Phase.PROPOSE,
            text="I argue for B.\n\nAnswer: B",
            token_count=12,
            slot_id=1,
            fields={"answer": "B"},
        ),
    )


def _make_builder_with_mock_envs(
    *,
    episode_log_dir: str | None,
    trained_role: Role = Role.DEBATER_A,
    with_thinking: bool = False,
    outcome: DebateOutcome | None = None,
    selfplay: bool = False,
) -> tuple[DebateGroupBuilder, list[DebateEnv]]:
    """Create a DebateGroupBuilder and mock DebateEnv(s) for testing on_group_complete."""
    if with_thinking:
        transcript = (
            Utterance(
                role=trained_role,
                round_index=0,
                phase=Phase.PROPOSE,
                text="<thinking>The answer is B</thinking>I argue for A.\n\nAnswer: A",
                token_count=20,
                slot_id=0,
                fields={"answer": "A"},
            ),
            Utterance(
                role=Role.DEBATER_B if trained_role == Role.DEBATER_A else Role.DEBATER_A,
                round_index=0,
                phase=Phase.PROPOSE,
                text="I argue for B.\n\nAnswer: B",
                token_count=12,
                slot_id=1,
                fields={"answer": "B"},
            ),
        )
    else:
        transcript = _make_utterances(trained_role)

    state = _make_state(trained_role=trained_role, transcript=transcript, outcome=outcome)
    runtime = DebateRuntime(state)

    renderer = MagicMock()
    builder = DebateGroupBuilder(
        problem=DebateProblemSpec.from_seat_answers(
            "What is 2+2?",
            "A",
            "B",
            ScoringMode.MCQ,
        ),
        game=DebateGameSpec(ProtocolKind.SEQUENTIAL, num_rounds=1),
        renderer=renderer,
        episode_log_dir=episode_log_dir,
        opponent_completer=None if selfplay else MagicMock(),
        group_size=1,
    )
    builder._runtimes = [runtime]

    opponent_role = Role.DEBATER_B if trained_role == Role.DEBATER_A else Role.DEBATER_A
    env = DebateEnv(
        role=trained_role,
        runtime=runtime,
        renderer=renderer,
        opponent_role=opponent_role,
    )
    return builder, [env]


# ---------------------------------------------------------------------------
# Test 1: Unit test for on_group_complete with mock debate envs
# ---------------------------------------------------------------------------


class TestOnGroupComplete:
    def test_writes_jsonl_record(self):
        """on_group_complete writes a valid JSONL record with expected fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder, envs = _make_builder_with_mock_envs(episode_log_dir=tmpdir)
            metrics = {"id/accuracy.trained": 1.0, "id/win_rate.trained": 1.0}

            builder.on_group_complete(
                trajectories_G=[],
                env_group=envs,
                rewards_and_metrics_G=[(1.0, metrics)],
            )

            log_path = os.path.join(tmpdir, "episodes.jsonl")
            assert os.path.exists(log_path)

            with open(log_path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["trained_role"] == "debater_a"
            assert record["reward_trained"] == 1.0
            assert len(record["transcript"]) == 2

    def test_multiple_envs_write_multiple_records(self):
        """Multiple envs in the group produce one record per env."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder, envs_a = _make_builder_with_mock_envs(
                episode_log_dir=tmpdir, trained_role=Role.DEBATER_A
            )
            # Add a second env (debater_b).
            _, envs_b = _make_builder_with_mock_envs(
                episode_log_dir=tmpdir, trained_role=Role.DEBATER_B
            )
            # Merge runtimes and envs into the first builder.
            builder._runtimes.extend([envs_b[0].runtime])
            all_envs = envs_a + envs_b

            builder.on_group_complete(
                trajectories_G=[],
                env_group=all_envs,
                rewards_and_metrics_G=[
                    (1.0, {"id/win_rate.trained": 1.0}),
                    (-1.0, {"id/win_rate.trained": 0.0}),
                ],
            )

            log_path = os.path.join(tmpdir, "episodes.jsonl")
            with open(log_path) as f:
                lines = f.readlines()
            assert len(lines) == 2
            r0 = json.loads(lines[0])
            r1 = json.loads(lines[1])
            assert r0["trained_role"] == "debater_a"
            assert r1["trained_role"] == "debater_b"
            assert r0["reward_trained"] == 1.0
            assert r1["reward_trained"] == -1.0


# ---------------------------------------------------------------------------
# Test 2: Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def _get_record(self, *, with_thinking: bool = False, outcome: DebateOutcome | None = None):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder, envs = _make_builder_with_mock_envs(
                episode_log_dir=tmpdir,
                with_thinking=with_thinking,
                outcome=outcome,
            )
            metrics = {
                "id/accuracy.trained": 1.0,
                "id/win_rate.trained": 1.0,
                "accuracy.debater_a": 0.5,  # non-id/ key should be excluded
            }
            builder.on_group_complete(
                trajectories_G=[],
                env_group=envs,
                rewards_and_metrics_G=[(1.0, metrics)],
            )
            log_path = os.path.join(tmpdir, "episodes.jsonl")
            with open(log_path) as f:
                return json.loads(f.readline())

    def test_schema_version(self):
        record = self._get_record()
        assert record["schema_version"] == 1

    def test_trained_role_present_and_valid(self):
        record = self._get_record()
        assert record["trained_role"] in ("debater_a", "debater_b")

    def test_transcript_has_identity(self):
        record = self._get_record()
        transcript = record["transcript"]
        assert isinstance(transcript, list)
        assert len(transcript) > 0
        for entry in transcript:
            assert "identity" in entry
            assert entry["identity"] in ("trained", "opponent")

    def test_identity_mapping_matches_trained_role(self):
        """If trained_role=debater_a, utterances by debater_a have identity='trained'."""
        record = self._get_record()
        trained_role = record["trained_role"]
        for entry in record["transcript"]:
            if entry["role"] == trained_role:
                assert entry["identity"] == "trained"
            else:
                assert entry["identity"] == "opponent"

    def test_identity_mapping_debater_b(self):
        """Verify identity mapping when trained_role=debater_b."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder, envs = _make_builder_with_mock_envs(
                episode_log_dir=tmpdir, trained_role=Role.DEBATER_B
            )
            builder.on_group_complete(
                trajectories_G=[],
                env_group=envs,
                rewards_and_metrics_G=[(1.0, {"id/win_rate.trained": 1.0})],
            )
            log_path = os.path.join(tmpdir, "episodes.jsonl")
            with open(log_path) as f:
                record = json.loads(f.readline())

        assert record["trained_role"] == "debater_b"
        for entry in record["transcript"]:
            if entry["role"] == "debater_b":
                assert entry["identity"] == "trained"
            else:
                assert entry["identity"] == "opponent"

    def test_signals_only_id_prefixed_frozen_opp(self):
        """In frozen-opp mode, signals dict contains only id/ prefixed keys."""
        record = self._get_record()
        for key in record["signals"]:
            assert key.startswith("id/"), f"Signal key {key!r} missing 'id/' prefix"

    def test_valid_jsonl(self):
        """Each line is valid JSON (JSONL format)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder, envs = _make_builder_with_mock_envs(episode_log_dir=tmpdir)
            # Write twice to get multiple lines.
            builder.on_group_complete(
                trajectories_G=[],
                env_group=envs,
                rewards_and_metrics_G=[(1.0, {})],
            )
            builder.on_group_complete(
                trajectories_G=[],
                env_group=envs,
                rewards_and_metrics_G=[(0.5, {})],
            )
            log_path = os.path.join(tmpdir, "episodes.jsonl")
            with open(log_path) as f:
                lines = f.readlines()
            assert len(lines) == 2
            for i, line in enumerate(lines):
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    pytest.fail(f"Line {i} is not valid JSON: {line!r}")

    def test_transcript_entry_fields(self):
        """Each transcript entry has role, phase, round, text, identity."""
        record = self._get_record()
        for entry in record["transcript"]:
            assert "role" in entry
            assert "phase" in entry
            assert "round" in entry
            assert "text" in entry
            assert "identity" in entry

    def test_outcome_fields(self):
        """Winner and verdict_text are populated from outcome."""
        outcome = DebateOutcome(
            winner=Role.DEBATER_A,
            scores_by_role={Role.DEBATER_A: 1.0, Role.DEBATER_B: -1.0},
            verdict_text="A wins because they are correct.",
        )
        record = self._get_record(outcome=outcome)
        assert record["winner"] == "debater_a"
        assert record["verdict_text"] == "A wins because they are correct."


# ---------------------------------------------------------------------------
# Test 3: Self-play signals
# ---------------------------------------------------------------------------


class TestSelfPlaySignals:
    def test_selfplay_signals_nonempty(self):
        """In self-play mode, signals includes all metrics (no id/ filter)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder, envs = _make_builder_with_mock_envs(episode_log_dir=tmpdir, selfplay=True)
            metrics = {
                "accuracy.debater_a": 1.0,
                "win_rate.debater_b": 0.0,
                "judge_confidence": 0.95,
            }
            builder.on_group_complete(
                trajectories_G=[],
                env_group=envs,
                rewards_and_metrics_G=[(1.0, metrics)],
            )
            log_path = os.path.join(tmpdir, "episodes.jsonl")
            with open(log_path) as f:
                record = json.loads(f.readline())

        assert record["signals"] == metrics

    def test_selfplay_schema_v2_seat_based(self):
        """Self-play records use schema_version=2 with seat-based fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder, envs = _make_builder_with_mock_envs(episode_log_dir=tmpdir, selfplay=True)
            builder.on_group_complete(
                trajectories_G=[],
                env_group=envs,
                rewards_and_metrics_G=[(1.0, {"accuracy.debater_a": 0.5})],
            )
            log_path = os.path.join(tmpdir, "episodes.jsonl")
            with open(log_path) as f:
                record = json.loads(f.readline())

        assert record["schema_version"] == 2
        # Seat-based fields.
        assert "role" in record
        assert "reward" in record
        assert record["role"] in ("debater_a", "debater_b")
        # No identity-framed fields.
        assert "trained_role" not in record
        assert "reward_trained" not in record
        # Answers keyed by seat.
        assert "public_debater_a" in record["answers"]
        assert "public_debater_b" in record["answers"]
        assert "public_trained" not in record["answers"]

    def test_selfplay_transcript_identity_uses_seat(self):
        """In self-play mode, transcript identity uses seat names (debater_a/debater_b)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder, envs = _make_builder_with_mock_envs(episode_log_dir=tmpdir, selfplay=True)
            builder.on_group_complete(
                trajectories_G=[],
                env_group=envs,
                rewards_and_metrics_G=[(1.0, {})],
            )
            log_path = os.path.join(tmpdir, "episodes.jsonl")
            with open(log_path) as f:
                record = json.loads(f.readline())

        for entry in record["transcript"]:
            assert entry["identity"] in ("debater_a", "debater_b")
            assert entry["identity"] == entry["role"]

    def test_frozen_opp_schema_v1_identity_based(self):
        """Frozen-opp records use schema_version=1 with identity-based fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder, envs = _make_builder_with_mock_envs(episode_log_dir=tmpdir, selfplay=False)
            builder.on_group_complete(
                trajectories_G=[],
                env_group=envs,
                rewards_and_metrics_G=[(1.0, {"id/accuracy.trained": 1.0})],
            )
            log_path = os.path.join(tmpdir, "episodes.jsonl")
            with open(log_path) as f:
                record = json.loads(f.readline())

        assert record["schema_version"] == 1
        assert "trained_role" in record
        assert "reward_trained" in record
        assert "role" not in record
        assert "public_trained" in record["answers"]
        assert "public_opponent" in record["answers"]

    def test_frozen_opp_signals_filters_id_prefix(self):
        """In frozen-opp mode, signals only contains id/-prefixed metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder, envs = _make_builder_with_mock_envs(episode_log_dir=tmpdir, selfplay=False)
            metrics = {
                "id/accuracy.trained": 1.0,
                "accuracy.debater_a": 0.5,  # should be excluded
            }
            builder.on_group_complete(
                trajectories_G=[],
                env_group=envs,
                rewards_and_metrics_G=[(1.0, metrics)],
            )
            log_path = os.path.join(tmpdir, "episodes.jsonl")
            with open(log_path) as f:
                record = json.loads(f.readline())

        assert "id/accuracy.trained" in record["signals"]
        assert "accuracy.debater_a" not in record["signals"]


# ---------------------------------------------------------------------------
# Test 4: File lock concurrency test
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_writes_no_corruption(self):
        """Writing 2 records from 2 threads produces 2 intact lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder1, envs1 = _make_builder_with_mock_envs(episode_log_dir=tmpdir)
            builder2, envs2 = _make_builder_with_mock_envs(episode_log_dir=tmpdir)

            errors: list[Exception] = []

            def _write(builder, envs):
                try:
                    builder.on_group_complete(
                        trajectories_G=[],
                        env_group=envs,
                        rewards_and_metrics_G=[(1.0, {})],
                    )
                except Exception as e:
                    errors.append(e)

            t1 = threading.Thread(target=_write, args=(builder1, envs1))
            t2 = threading.Thread(target=_write, args=(builder2, envs2))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            assert not errors, f"Threads raised: {errors}"

            log_path = os.path.join(tmpdir, "episodes.jsonl")
            with open(log_path) as f:
                lines = f.readlines()

            assert len(lines) == 2
            for i, line in enumerate(lines):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    pytest.fail(f"Line {i} is corrupt/interleaved: {line!r}")
                assert record["schema_version"] == 1

    def test_lock_is_module_level(self):
        """_EPISODE_LOG_LOCK is a module-level threading.Lock, shared across builders."""
        assert isinstance(_EPISODE_LOG_LOCK, type(threading.Lock()))


# ---------------------------------------------------------------------------
# Test 4: Wiring verification (CLIConfig -> DebateRLDatasetBuilder -> DebateDataset -> DebateGroupBuilder)
# ---------------------------------------------------------------------------


class TestWiring:
    def test_debate_dataset_passes_episode_log_dir(self):
        """DebateDataset passes episode_log_dir to DebateGroupBuilder in get_batch."""
        from tinker_cookbook.recipes.multiplayer_rl.debate.dataset import DebateDataset

        ds = DebateDataset(
            problems=[DebateProblemSpec.from_seat_answers("q1", "a1", "b1", ScoringMode.MCQ)],
            batch_size=1,
            group_size=1,
            game=DebateGameSpec(ProtocolKind.SEQUENTIAL, num_rounds=1),
            renderer=MagicMock(),
            episode_log_dir="/tmp/test-episodes",
        )
        batch = ds.get_batch(0)
        assert len(batch) == 1
        assert isinstance(batch[0], DebateGroupBuilder)
        assert batch[0].episode_log_dir == "/tmp/test-episodes"

    def test_debate_dataset_none_by_default(self):
        """episode_log_dir defaults to None when not specified."""
        from tinker_cookbook.recipes.multiplayer_rl.debate.dataset import DebateDataset

        ds = DebateDataset(
            problems=[DebateProblemSpec.from_seat_answers("q1", "a1", "b1", ScoringMode.MCQ)],
            batch_size=1,
            group_size=1,
            game=DebateGameSpec(ProtocolKind.SEQUENTIAL, num_rounds=1),
            renderer=MagicMock(),
        )
        batch = ds.get_batch(0)
        assert batch[0].episode_log_dir is None

    def test_cli_config_has_episode_log_dir(self):
        """CLIConfig has episode_log_dir field."""
        from tinker_cookbook.recipes.multiplayer_rl.debate.scripts.train import CLIConfig

        config = CLIConfig()
        assert hasattr(config, "episode_log_dir")
        assert config.episode_log_dir is None

    def test_dataset_builder_has_episode_log_dir(self):
        """DebateRLDatasetBuilder has episode_log_dir field."""
        from tinker_cookbook.recipes.multiplayer_rl.debate.scripts.train import (
            DebateRLDatasetBuilder,
        )

        # Check that the field exists on the class.
        import inspect

        sig = inspect.signature(DebateRLDatasetBuilder)
        assert "episode_log_dir" in sig.parameters

    def test_build_config_passes_episode_log_dir(self):
        """build_config wires episode_log_dir from CLIConfig through to DebateRLDatasetBuilder."""
        from tinker_cookbook.recipes.multiplayer_rl.debate.scripts.train import (
            CLIConfig,
            DebateRLDatasetBuilder,
            build_config,
        )

        cli = CLIConfig(episode_log_dir="/tmp/test-wire")
        config = build_config(cli)
        assert isinstance(config.dataset_builder, DebateRLDatasetBuilder)
        assert config.dataset_builder.episode_log_dir == "/tmp/test-wire"


# ---------------------------------------------------------------------------
# Test 5: No-op when episode_log_dir is None
# ---------------------------------------------------------------------------


class TestNoOp:
    def test_no_file_created_when_none(self):
        """on_group_complete is a no-op when episode_log_dir is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder, envs = _make_builder_with_mock_envs(episode_log_dir=None)

            builder.on_group_complete(
                trajectories_G=[],
                env_group=envs,
                rewards_and_metrics_G=[(1.0, {})],
            )

            # No file should be created anywhere.
            # (Using tmpdir here to verify nothing was written to a default location.)
            assert not os.listdir(tmpdir)

    def test_empty_env_group_creates_empty_file(self):
        """on_group_complete with empty env_group writes no records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DebateGroupBuilder(
                problem=DebateProblemSpec.from_seat_answers("Q", "A", "B", ScoringMode.MCQ),
                game=DebateGameSpec(ProtocolKind.SEQUENTIAL, num_rounds=1),
                renderer=MagicMock(),
                episode_log_dir=tmpdir,
            )

            builder.on_group_complete(
                trajectories_G=[],
                env_group=[],
                rewards_and_metrics_G=[],
            )

            episodes_path = os.path.join(tmpdir, "episodes.jsonl")
            assert os.path.exists(episodes_path)
            with open(episodes_path) as f:
                assert f.read() == ""
