"""Wave 2 stress tests — builder hierarchy, field parity, registry completeness."""

import pytest

from tinker_cookbook.recipes.rlvr.builders import (
    DATASET_BUILDER_MAP,
    DeepMathBuilder,
    GpqaOpenEndedBuilder,
    Gsm8kBuilder,
    MathBuilder,
    OmniMathBuilder,
    PolarisBuilder,
    RLVRDatasetBuilder,
    SympyBoxedBuilder,
    _standard_fewshot_prefix,
)
from tinker_cookbook.recipes.rlvr.env import ANSWER_FORMAT_INSTRUCTION, BOXED_FORMAT_INSTRUCTION
from tinker_cookbook.recipes.rlvr.graders import GraderConfig, LLMGraderConfig, SympyGraderConfig


# Shared kwargs needed to construct any builder (abstract fields with no defaults)
_COMMON = dict(
    batch_size=4,
    group_size=2,
    model_name_for_tokenizer="meta-llama/Llama-3.1-8B-Instruct",
    renderer_name="llama3",
)


# ===========================================================================
# (a) chz field override chain — grader_config inheritance
# ===========================================================================


class TestGraderConfigInheritance:
    def test_base_has_no_default(self):
        """RLVRDatasetBuilder.grader_config has no default — it's effectively abstract.

        We verify this by checking the chz field metadata: the base class declares
        grader_config without a default, so constructing a concrete subclass that
        doesn't override it should fail.
        """
        import chz

        # Create a minimal concrete subclass that does NOT provide grader_config
        @chz.chz
        class _BareBuilder(RLVRDatasetBuilder):
            def _load_data(self):
                return [], None

        # Should fail because grader_config has no default on the base
        with pytest.raises(TypeError):
            _BareBuilder(**_COMMON)

    def test_sympy_boxed_defaults_to_sympy_grader(self):
        """SympyBoxedBuilder.grader_config defaults to SympyGraderConfig()."""
        builder = MathBuilder(**_COMMON)
        assert isinstance(builder.grader_config, SympyGraderConfig)

    def test_math_inherits_sympy_grader(self):
        builder = MathBuilder(**_COMMON)
        assert isinstance(builder.grader_config, SympyGraderConfig)
        assert isinstance(builder.grader_config, GraderConfig)

    def test_gsm8k_inherits_sympy_grader(self):
        builder = Gsm8kBuilder(**_COMMON)
        assert isinstance(builder.grader_config, SympyGraderConfig)

    def test_polaris_inherits_sympy_grader(self):
        builder = PolarisBuilder(**_COMMON)
        assert isinstance(builder.grader_config, SympyGraderConfig)

    def test_deepmath_inherits_sympy_grader(self):
        builder = DeepMathBuilder(**_COMMON)
        assert isinstance(builder.grader_config, SympyGraderConfig)

    def test_gpqa_defaults_to_llm_grader(self):
        builder = GpqaOpenEndedBuilder(**_COMMON)
        assert isinstance(builder.grader_config, LLMGraderConfig)

    def test_omnimath_defaults_to_llm_grader(self):
        builder = OmniMathBuilder(**_COMMON)
        assert isinstance(builder.grader_config, LLMGraderConfig)


# ===========================================================================
# (b) format_instruction parity
# ===========================================================================


class TestFormatInstructionParity:
    def test_math_format(self):
        builder = MathBuilder(**_COMMON)
        assert builder.format_instruction == " Write your answer in \\boxed{} format."

    def test_gsm8k_format(self):
        builder = Gsm8kBuilder(**_COMMON)
        assert builder.format_instruction == " Provide a numerical answer without units, written inside \\boxed{}."

    def test_gpqa_inherits_answer_format_instruction(self):
        builder = GpqaOpenEndedBuilder(**_COMMON)
        assert builder.format_instruction == ANSWER_FORMAT_INSTRUCTION

    def test_omnimath_uses_boxed_format_instruction(self):
        builder = OmniMathBuilder(**_COMMON)
        assert builder.format_instruction == BOXED_FORMAT_INSTRUCTION

    def test_polaris_uses_boxed_format(self):
        """Polaris inherits from SympyBoxedBuilder, so gets _BOXED_FORMAT_INSTRUCTION."""
        builder = PolarisBuilder(**_COMMON)
        assert builder.format_instruction == " Write your answer in \\boxed{} format."

    def test_deepmath_uses_boxed_format(self):
        builder = DeepMathBuilder(**_COMMON)
        assert builder.format_instruction == " Write your answer in \\boxed{} format."

    def test_math_vs_gsm8k_differ(self):
        """Math and GSM8K must have different format instructions."""
        math_b = MathBuilder(**_COMMON)
        gsm_b = Gsm8kBuilder(**_COMMON)
        assert math_b.format_instruction != gsm_b.format_instruction

    def test_gpqa_vs_omnimath_differ(self):
        """GPQA (final_answer tags) vs OmniMath (boxed) use different format instructions."""
        gpqa_b = GpqaOpenEndedBuilder(**_COMMON)
        omni_b = OmniMathBuilder(**_COMMON)
        assert gpqa_b.format_instruction != omni_b.format_instruction


# ===========================================================================
# (c) seed parity
# ===========================================================================


class TestSeedParity:
    def test_sympy_boxed_subclasses_default_seed_0(self):
        for cls in [MathBuilder, Gsm8kBuilder, PolarisBuilder, DeepMathBuilder]:
            builder = cls(**_COMMON)
            assert builder.seed == 0, f"{cls.__name__}.seed should be 0, got {builder.seed}"

    def test_llm_graded_builders_default_seed_42(self):
        for cls in [GpqaOpenEndedBuilder, OmniMathBuilder]:
            builder = cls(**_COMMON)
            assert builder.seed == 42, f"{cls.__name__}.seed should be 42, got {builder.seed}"

    def test_seed_override_works(self):
        """Verify seed can be overridden via chz construction."""
        builder = MathBuilder(**_COMMON, seed=123)
        assert builder.seed == 123


# ===========================================================================
# (d) _get_extract_fn parity
# ===========================================================================


class TestExtractFnParity:
    def test_sympy_boxed_builders_use_extract_boxed(self):
        from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

        for cls in [MathBuilder, Gsm8kBuilder, PolarisBuilder, DeepMathBuilder]:
            builder = cls(**_COMMON)
            fn = builder._get_extract_fn()
            assert fn is extract_boxed, f"{cls.__name__}._get_extract_fn() should return extract_boxed"

    def test_omnimath_uses_extract_boxed(self):
        from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

        builder = OmniMathBuilder(**_COMMON)
        fn = builder._get_extract_fn()
        assert fn is extract_boxed

    def test_gpqa_uses_extract_final_answer(self):
        from tinker_cookbook.recipes.rlvr.env import extract_final_answer

        builder = GpqaOpenEndedBuilder(**_COMMON)
        fn = builder._get_extract_fn()
        assert fn is extract_final_answer

    def test_extract_fn_callable(self):
        """All extract functions should raise ValueError on bad input."""
        for cls in [MathBuilder, GpqaOpenEndedBuilder, OmniMathBuilder]:
            builder = cls(**_COMMON)
            fn = builder._get_extract_fn()
            with pytest.raises(ValueError):
                fn("no valid content here")


# ===========================================================================
# (e) include_fewshot behavior
# ===========================================================================


class TestFewshotBehavior:
    def test_math_fewshot_true_returns_prefix(self):
        builder = MathBuilder(**_COMMON, include_fewshot=True)
        prefix = builder._resolve_convo_prefix()
        assert prefix is not None
        assert len(prefix) == 2
        assert prefix[0]["role"] == "user"
        assert prefix[1]["role"] == "assistant"
        assert "strawberry" in prefix[0]["content"]
        assert "\\boxed{3}" in prefix[1]["content"]

    def test_math_fewshot_false_returns_none(self):
        builder = MathBuilder(**_COMMON, include_fewshot=False)
        prefix = builder._resolve_convo_prefix()
        assert prefix is None

    def test_polaris_default_no_fewshot(self):
        builder = PolarisBuilder(**_COMMON)
        assert builder.include_fewshot is False
        prefix = builder._resolve_convo_prefix()
        assert prefix is None

    def test_gpqa_no_fewshot(self):
        """GpqaOpenEndedBuilder inherits from RLVRDatasetBuilder (no include_fewshot field)."""
        builder = GpqaOpenEndedBuilder(**_COMMON)
        prefix = builder._resolve_convo_prefix()
        assert prefix is None

    def test_omnimath_no_fewshot(self):
        builder = OmniMathBuilder(**_COMMON)
        prefix = builder._resolve_convo_prefix()
        assert prefix is None

    def test_math_default_has_fewshot(self):
        """MathBuilder default is include_fewshot=True (inherited from SympyBoxedBuilder)."""
        builder = MathBuilder(**_COMMON)
        assert builder.include_fewshot is True
        prefix = builder._resolve_convo_prefix()
        assert prefix is not None


# ===========================================================================
# (f) eval group_size=1 — verified by code inspection
# ===========================================================================


class TestEvalGroupSize:
    def test_eval_dataset_gets_group_size_1(self):
        """Verify __call__ constructs eval_ds with group_size=1 by inspecting source.

        We can't call __call__ without network (HF dataset downloads), so we verify
        the code path. The RLVRDatasetBuilder.__call__ method passes group_size=1
        to the eval RLVRDataset constructor.
        """
        import ast
        import inspect
        import textwrap

        source = textwrap.dedent(inspect.getsource(RLVRDatasetBuilder.__call__))
        tree = ast.parse(source)

        # Find all keyword arguments named 'group_size' with value 1
        found_group_size_1 = False
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword):
                if node.arg == "group_size" and isinstance(node.value, ast.Constant):
                    if node.value.value == 1:
                        found_group_size_1 = True

        assert found_group_size_1, "eval dataset should be constructed with group_size=1"

    def test_train_dataset_gets_configured_group_size(self):
        """Verify train dataset uses self.group_size (not hardcoded)."""
        import ast
        import inspect
        import textwrap

        source = textwrap.dedent(inspect.getsource(RLVRDatasetBuilder.__call__))
        tree = ast.parse(source)

        # Find keyword arguments named 'group_size' with value self.group_size
        found_self_group_size = False
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword):
                if node.arg == "group_size" and isinstance(node.value, ast.Attribute):
                    if node.value.attr == "group_size":
                        found_self_group_size = True

        assert found_self_group_size, "train dataset should use self.group_size"


# ===========================================================================
# (g) DATASET_BUILDER_MAP completeness
# ===========================================================================


class TestBuilderMapCompleteness:
    EXPECTED_KEYS = {"math", "gsm8k", "polaris", "deepmath", "gpqa_oe", "omni_math"}

    def test_all_keys_present(self):
        assert set(DATASET_BUILDER_MAP.keys()) == self.EXPECTED_KEYS

    def test_all_values_are_subclasses(self):
        for key, cls in DATASET_BUILDER_MAP.items():
            assert issubclass(cls, RLVRDatasetBuilder), f"{key} -> {cls} is not an RLVRDatasetBuilder subclass"

    def test_values_are_classes_not_instances(self):
        for key, val in DATASET_BUILDER_MAP.items():
            assert isinstance(val, type), f"{key} should map to a class, got {type(val)}"

    def test_no_extra_keys(self):
        assert len(DATASET_BUILDER_MAP) == 6

    def test_map_matches_expected_classes(self):
        expected = {
            "math": MathBuilder,
            "gsm8k": Gsm8kBuilder,
            "polaris": PolarisBuilder,
            "deepmath": DeepMathBuilder,
            "gpqa_oe": GpqaOpenEndedBuilder,
            "omni_math": OmniMathBuilder,
        }
        assert DATASET_BUILDER_MAP == expected


# ===========================================================================
# (h) _standard_fewshot_prefix content
# ===========================================================================


class TestFewshotPrefixParity:
    def test_structure(self):
        prefix = _standard_fewshot_prefix()
        assert len(prefix) == 2
        assert prefix[0]["role"] == "user"
        assert prefix[1]["role"] == "assistant"

    def test_expected_content(self):
        """Verify fewshot prefix contains the strawberry-counting pair."""
        prefix = _standard_fewshot_prefix()
        assert "strawberry" in prefix[0]["content"]
        assert "\\boxed{3}" in prefix[1]["content"]
