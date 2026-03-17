"""Wave 3 stress tests: CLI, math_rl wrapper, import hygiene."""

import inspect

import pytest

from tinker_cookbook.recipes.math_rl import arithmetic_env
from tinker_cookbook.recipes.math_rl.train import get_dataset_builder
from tinker_cookbook.recipes.rlvr.builders import (
    DATASET_BUILDER_MAP,
    Gsm8kBuilder,
    MathBuilder,
    RLVRDatasetBuilder,
)
from tinker_cookbook.recipes.rlvr.train import CLIConfig, derive_loss_fn_config
from tinker_cookbook.rl.train import Config

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _make_cli(**overrides) -> CLIConfig:
    """Build a CLIConfig with sensible defaults; override any field via kwargs."""
    defaults = dict(
        model_name=MODEL,
        dataset="math",
        batch_size=4,
        group_size=4,
        max_tokens=512,
    )
    defaults.update(overrides)
    return CLIConfig(**defaults)


# ---------------------------------------------------------------------------
# (a) CLI required field enforcement
# ---------------------------------------------------------------------------


class TestCLIRequiredFields:
    def test_missing_model_name_raises(self):
        with pytest.raises(TypeError):
            CLIConfig(
                dataset="math",
                batch_size=4,
                group_size=4,
                max_tokens=512,
                advantage_scheme="grpo",
            )

    def test_missing_dataset_raises(self):
        with pytest.raises(TypeError):
            CLIConfig(
                model_name=MODEL,
                batch_size=4,
                group_size=4,
                max_tokens=512,
                advantage_scheme="grpo",
            )

    def test_missing_batch_size_raises(self):
        with pytest.raises(TypeError):
            CLIConfig(
                model_name=MODEL,
                dataset="math",
                group_size=4,
                max_tokens=512,
                advantage_scheme="grpo",
            )

    def test_advantage_scheme_defaults_to_maxrl(self):
        cfg = _make_cli()
        assert cfg.advantage_scheme == "maxrl"

    def test_v3b_defaults(self):
        """CLI defaults match v3b production values."""
        cfg = _make_cli()
        assert cfg.loss_fn == "ppo"
        assert cfg.grad_clip_norm == 0.3
        assert cfg.format_coef == 0.1
        assert cfg.eos_coef == 0.1
        assert cfg.clip_ratio_lower == 0.2
        assert cfg.clip_ratio_upper == 0.2

    def test_all_required_succeeds(self):
        cfg = _make_cli(advantage_scheme="grpo")
        assert cfg.model_name == MODEL
        assert cfg.dataset == "math"

    def test_exactly_five_required(self):
        """The 5 fields without defaults are exactly the required ones."""
        sig = inspect.signature(CLIConfig)
        required = {
            name
            for name, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty and not name.startswith("__")
        }
        assert required == {
            "model_name",
            "dataset",
            "batch_size",
            "group_size",
            "max_tokens",
        }


# ---------------------------------------------------------------------------
# (b) math_rl wrapper routing
# ---------------------------------------------------------------------------


class TestMathRLWrapperRouting:
    def test_arithmetic_returns_arithmetic_builder(self):
        builder = get_dataset_builder(
            env="arithmetic",
            batch_size=4,
            model_name=MODEL,
            renderer_name="llama3",
            group_size=4,
        )
        assert isinstance(builder, arithmetic_env.ArithmeticDatasetBuilder)

    def test_math_routes_to_rlvr(self):
        builder = get_dataset_builder(
            env="math",
            batch_size=4,
            model_name=MODEL,
            renderer_name="llama3",
            group_size=4,
        )
        assert isinstance(builder, MathBuilder)

    def test_gsm8k_routes_to_rlvr(self):
        builder = get_dataset_builder(
            env="gsm8k",
            batch_size=4,
            model_name=MODEL,
            renderer_name="llama3",
            group_size=4,
        )
        assert isinstance(builder, Gsm8kBuilder)

    def test_unknown_env_raises(self):
        with pytest.raises(ValueError, match="Unknown environment"):
            get_dataset_builder(
                env="bogus",
                batch_size=4,
                model_name=MODEL,
                renderer_name="llama3",
                group_size=4,
            )


# ---------------------------------------------------------------------------
# (c) DATASET_BUILDER_MAP type check
# ---------------------------------------------------------------------------


class TestBuilderMapTypes:
    def test_all_values_are_rlvr_builder_subclasses(self):
        for key, cls in DATASET_BUILDER_MAP.items():
            assert issubclass(cls, RLVRDatasetBuilder), (
                f"{key} -> {cls} is not a subclass of RLVRDatasetBuilder"
            )

    def test_values_are_classes_not_instances(self):
        for key, cls in DATASET_BUILDER_MAP.items():
            assert isinstance(cls, type), f"{key} -> {cls} should be a class, not an instance"


# ---------------------------------------------------------------------------
# (d) No circular imports
# ---------------------------------------------------------------------------


class TestNoCircularImports:
    """Import the modules in every permutation pair to surface cycles."""

    MODULES = [
        "tinker_cookbook.recipes.rlvr.train",
        "tinker_cookbook.recipes.rlvr.builders",
        "tinker_cookbook.recipes.rlvr.graders",
        "tinker_cookbook.recipes.rlvr.env",
        "tinker_cookbook.recipes.rlvr.types",
    ]

    def test_import_all_at_once(self):
        import importlib

        for mod_name in self.MODULES:
            importlib.import_module(mod_name)

    def test_import_reverse_order(self):
        import importlib
        import sys

        cached = {}
        for mod_name in self.MODULES:
            if mod_name in sys.modules:
                cached[mod_name] = sys.modules.pop(mod_name)

        try:
            for mod_name in reversed(self.MODULES):
                importlib.import_module(mod_name)
        finally:
            sys.modules.update(cached)


# ---------------------------------------------------------------------------
# (e) CLI Config fields match rl.train.Config
# ---------------------------------------------------------------------------


CONFIG_FIELDS_USED_BY_CLI_MAIN = {
    "learning_rate",
    "dataset_builder",
    "model_name",
    "renderer_name",
    "lora_rank",
    "max_tokens",
    "temperature",
    "eval_temperature",
    "eval_top_p",
    "advantage_scheme",
    "wandb_project",
    "wandb_name",
    "log_path",
    "base_url",
    "load_checkpoint_path",
    "eval_every",
    "eval_on_start",
    "save_every",
    "sampling_max_connections",
    "kl_penalty_coef",
    "kl_reference_config",
    "compute_post_kl",
    "num_substeps",
    "loss_fn",
    "loss_fn_config",
    "grad_clip_norm",
    "remove_constant_reward_groups",
    "async_config",
    "stream_minibatch_config",
}


class TestCLIConfigFieldsMatchRLConfig:
    """Verify every kwarg that cli_main passes to Config actually exists on Config."""

    def test_all_cli_kwargs_exist_on_config(self):
        config_params = {
            name for name in inspect.signature(Config).parameters
            if not name.startswith("__")
        }
        for field_name in CONFIG_FIELDS_USED_BY_CLI_MAIN:
            assert field_name in config_params, (
                f"cli_main passes '{field_name}' to Config but Config has no such field"
            )

    def test_config_has_no_surprise_required_fields_uncovered(self):
        """Config's required fields (no default) should all be set by cli_main."""
        required = {
            name
            for name, p in inspect.signature(Config).parameters.items()
            if p.default is inspect.Parameter.empty and not name.startswith("__")
        }
        uncovered = required - CONFIG_FIELDS_USED_BY_CLI_MAIN
        assert not uncovered, f"Config requires {uncovered} but cli_main doesn't pass them"


# ---------------------------------------------------------------------------
# (f) DAPO-style clip ratio wiring
# ---------------------------------------------------------------------------


class TestClipRatioWiring:
    """Verify derive_loss_fn_config translates clip ratios to thresholds."""

    def test_default_symmetric_clip(self):
        """Default 0.2/0.2 -> thresholds 0.8/1.2."""
        cfg = _make_cli()
        assert cfg.loss_fn == "ppo"
        result = derive_loss_fn_config(cfg)
        assert result == {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}

    def test_dapo_asymmetric_clip(self):
        """DAPO uses 0.2 lower / 0.28 upper -> thresholds 0.8/1.28."""
        cfg = _make_cli(clip_ratio_lower=0.2, clip_ratio_upper=0.28)
        result = derive_loss_fn_config(cfg)
        assert result["clip_low_threshold"] == pytest.approx(0.8)
        assert result["clip_high_threshold"] == pytest.approx(1.28)

    def test_explicit_loss_fn_config_wins(self):
        """If user passes loss_fn_config directly, clip ratios are ignored."""
        cfg = _make_cli(loss_fn_config={"clip_low_threshold": 0.5, "clip_high_threshold": 1.5})
        result = derive_loss_fn_config(cfg)
        assert result == {"clip_low_threshold": 0.5, "clip_high_threshold": 1.5}

    def test_non_ppo_skips_clip(self):
        """When loss_fn is not ppo/cispo, clip ratios don't generate config."""
        cfg = _make_cli(loss_fn="importance_sampling")
        result = derive_loss_fn_config(cfg)
        assert result is None
