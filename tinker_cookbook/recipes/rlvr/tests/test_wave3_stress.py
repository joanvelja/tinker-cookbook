"""Wave 3 stress tests: CLI, math_rl wrapper, import hygiene."""

import inspect

import pytest


# ---------------------------------------------------------------------------
# (a) CLI required field enforcement
# ---------------------------------------------------------------------------


class TestCLIRequiredFields:
    def test_missing_model_name_raises(self):
        from tinker_cookbook.recipes.rlvr.train import CLIConfig

        with pytest.raises(TypeError):
            CLIConfig(
                dataset="math",
                batch_size=4,
                group_size=4,
                max_tokens=512,
                advantage_scheme="grpo",
            )

    def test_missing_dataset_raises(self):
        from tinker_cookbook.recipes.rlvr.train import CLIConfig

        with pytest.raises(TypeError):
            CLIConfig(
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                batch_size=4,
                group_size=4,
                max_tokens=512,
                advantage_scheme="grpo",
            )

    def test_missing_batch_size_raises(self):
        from tinker_cookbook.recipes.rlvr.train import CLIConfig

        with pytest.raises(TypeError):
            CLIConfig(
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                dataset="math",
                group_size=4,
                max_tokens=512,
                advantage_scheme="grpo",
            )

    def test_advantage_scheme_defaults_to_maxrl(self):
        from tinker_cookbook.recipes.rlvr.train import CLIConfig

        cfg = CLIConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            dataset="math",
            batch_size=4,
            group_size=4,
            max_tokens=512,
        )
        assert cfg.advantage_scheme == "maxrl"

    def test_all_required_succeeds(self):
        from tinker_cookbook.recipes.rlvr.train import CLIConfig

        cfg = CLIConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            dataset="math",
            batch_size=4,
            group_size=4,
            max_tokens=512,
            advantage_scheme="grpo",
        )
        assert cfg.model_name == "meta-llama/Llama-3.1-8B-Instruct"
        assert cfg.dataset == "math"

    def test_exactly_six_required(self):
        """The 6 fields without defaults are exactly the required ones."""
        from tinker_cookbook.recipes.rlvr.train import CLIConfig

        sig = inspect.signature(CLIConfig)
        required = [
            name
            for name, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty and not name.startswith("__")
        ]
        assert set(required) == {
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
        from tinker_cookbook.recipes.math_rl import arithmetic_env
        from tinker_cookbook.recipes.math_rl.train import get_dataset_builder

        builder = get_dataset_builder(
            env="arithmetic",
            batch_size=4,
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            renderer_name="llama3",
            group_size=4,
        )
        assert isinstance(builder, arithmetic_env.ArithmeticDatasetBuilder)

    def test_math_routes_to_rlvr(self):
        from tinker_cookbook.recipes.math_rl.train import get_dataset_builder
        from tinker_cookbook.recipes.rlvr.builders import MathBuilder

        builder = get_dataset_builder(
            env="math",
            batch_size=4,
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            renderer_name="llama3",
            group_size=4,
        )
        assert isinstance(builder, MathBuilder)

    def test_gsm8k_routes_to_rlvr(self):
        from tinker_cookbook.recipes.math_rl.train import get_dataset_builder
        from tinker_cookbook.recipes.rlvr.builders import Gsm8kBuilder

        builder = get_dataset_builder(
            env="gsm8k",
            batch_size=4,
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            renderer_name="llama3",
            group_size=4,
        )
        assert isinstance(builder, Gsm8kBuilder)

    def test_unknown_env_raises(self):
        from tinker_cookbook.recipes.math_rl.train import get_dataset_builder

        with pytest.raises(ValueError, match="Unknown environment"):
            get_dataset_builder(
                env="bogus",
                batch_size=4,
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                renderer_name="llama3",
                group_size=4,
            )


# ---------------------------------------------------------------------------
# (c) DATASET_BUILDER_MAP type check
# ---------------------------------------------------------------------------


class TestBuilderMapTypes:
    def test_all_values_are_rlvr_builder_subclasses(self):
        from tinker_cookbook.recipes.rlvr.builders import (
            DATASET_BUILDER_MAP,
            RLVRDatasetBuilder,
        )

        for key, cls in DATASET_BUILDER_MAP.items():
            assert issubclass(cls, RLVRDatasetBuilder), (
                f"{key} -> {cls} is not a subclass of RLVRDatasetBuilder"
            )

    def test_values_are_classes_not_instances(self):
        from tinker_cookbook.recipes.rlvr.builders import DATASET_BUILDER_MAP

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

        # Clear cached modules to test fresh import order
        cached = {}
        for mod_name in self.MODULES:
            if mod_name in sys.modules:
                cached[mod_name] = sys.modules.pop(mod_name)

        try:
            for mod_name in reversed(self.MODULES):
                importlib.import_module(mod_name)
        finally:
            # Restore
            sys.modules.update(cached)


# ---------------------------------------------------------------------------
# (e) CLI Config fields match rl.train.Config
# ---------------------------------------------------------------------------


class TestCLIConfigFieldsMatchRLConfig:
    """Verify every kwarg that cli_main passes to Config actually exists on Config."""

    CONFIG_FIELDS_USED_BY_CLI_MAIN = [
        "learning_rate",
        "dataset_builder",
        "model_name",
        "renderer_name",
        "lora_rank",
        "max_tokens",
        "temperature",
        "advantage_scheme",
        "wandb_project",
        "wandb_name",
        "log_path",
        "base_url",
        "load_checkpoint_path",
        "eval_every",
        "save_every",
        "sampling_max_connections",
        "kl_penalty_coef",
        "compute_post_kl",
        "num_substeps",
        "loss_fn",
        "loss_fn_config",
        "async_config",
    ]

    def test_all_cli_kwargs_exist_on_config(self):
        from tinker_cookbook.rl.train import Config

        sig = inspect.signature(Config)
        config_params = {
            name for name in sig.parameters if not name.startswith("__")
        }
        for field_name in self.CONFIG_FIELDS_USED_BY_CLI_MAIN:
            assert field_name in config_params, (
                f"cli_main passes '{field_name}' to Config but Config has no such field"
            )

    def test_config_has_no_surprise_required_fields_uncovered(self):
        """Config's required fields (no default) should all be set by cli_main."""
        from tinker_cookbook.rl.train import Config

        sig = inspect.signature(Config)
        required = {
            name
            for name, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty and not name.startswith("__")
        }
        covered = set(self.CONFIG_FIELDS_USED_BY_CLI_MAIN)
        uncovered = required - covered
        assert not uncovered, f"Config requires {uncovered} but cli_main doesn't pass them"
