"""Consultancy dataset builders.

Same as RLVR but the grader is reference-blind: the frozen judge sees only
(question, model_answer) and decides CORRECT/INCORRECT from its own knowledge.
The reference answer is still stored and logged for offline FN/FP analysis.
"""

import chz

from tinker_cookbook.llm_client import LLMClientConfig
from tinker_cookbook.recipes.rlvr.builders import RLVRDatasetBuilder
from tinker_cookbook.recipes.rlvr.graders import GraderConfig, LLMGraderConfig
from tinker_cookbook.recipes.rlvr.types import RLVRExample

# ---------------------------------------------------------------------------
# Reference-blind grader prompt
# ---------------------------------------------------------------------------

_CONSULTANCY_SYSTEM = """\
Evaluate the following argument on the basis of:
1. Is the reasoning chain logically sound? Are claims justified?
2. Are there unstated assumptions or logical gaps?
3. Does the evidence actually support the chosen answer?
Do NOT guess the correct answer yourself. Judge only by the quality of the argument and its reasoning.

CORRECT = the answer correctly addresses the question.
INCORRECT = the answer is wrong, incomplete, or nonsensical."""

_CONSULTANCY_USER = "Question: {question}\nAnswer: {response}"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


@chz.chz
class ConsultancyGpqaBuilder(RLVRDatasetBuilder):
    """GPQA with a reference-blind frozen-model judge.

    Identical to GpqaOpenEndedBuilder except:
    - The grader prompt omits the reference answer ({target} unused).
    - The default grader model + base_url point at the frozen base model.
    """

    grader_config: GraderConfig = LLMGraderConfig(
        client=LLMClientConfig(
            model="qwen/qwen3.5-35b-a3b",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            reasoning_effort="none",
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            presence_penalty=1.5,
            max_tokens=2048,
            extra_body={"provider": {"order": ["alibaba"]}},
        ),
        system_prompt=_CONSULTANCY_SYSTEM,
        user_template=_CONSULTANCY_USER,
        decision_tag="decision",
    )
    subset: str = "extended"
    split: str = "train"
    eval_frac: float = 0.1
    dataset_name: str = "consultancy_gpqa"

    def _load_data(self) -> tuple[list[RLVRExample], list[RLVRExample] | None]:
        from tinker_cookbook.recipes.multiplayer_rl.debate.data.gpqa import (
            load_gpqa_open_ended_rows,
        )

        raw = load_gpqa_open_ended_rows(
            subset=self.subset, split=self.split, seed=self.seed,
        )
        rows = [
            RLVRExample(question=str(r["question"]), reference=str(r["answer"]))
            for r in raw
        ]

        n_eval = max(1, int(len(rows) * self.eval_frac))
        train = rows[n_eval:]
        eval_ = rows[:n_eval]
        return train, eval_


# ---------------------------------------------------------------------------
# Builder registry
# ---------------------------------------------------------------------------

DATASET_BUILDER_MAP: dict[str, type[RLVRDatasetBuilder]] = {
    "consultancy_gpqa": ConsultancyGpqaBuilder,
}
