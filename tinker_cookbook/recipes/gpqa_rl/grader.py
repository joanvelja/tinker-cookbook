import asyncio
import logging

import chz

from tinker_cookbook.llm_client import AsyncLLMClient, LLMClientConfig

logger = logging.getLogger(__name__)


class GradingError(Exception):
    pass


class AmbiguousVerdictError(GradingError):
    pass


_DEFAULT_SYSTEM = (
    "You are a grading assistant. Compare the student's response to the target answer. "
    "Respond with exactly one word: CORRECT or INCORRECT."
)

_DEFAULT_USER = (
    "Question: {question}\n\nTarget answer: {target}\n\nStudent response: {response}\n\nVerdict:"
)


@chz.chz
class LLMGraderConfig:
    client: LLMClientConfig = LLMClientConfig()
    system_prompt: str = _DEFAULT_SYSTEM
    user_template: str = _DEFAULT_USER


class AsyncLLMGrader:
    def __init__(self, config: LLMGraderConfig) -> None:
        self.config = config
        self.client = AsyncLLMClient(config.client)

    async def grade(self, question: str, target: str, response: str) -> bool:
        user_prompt = self.config.user_template.format(
            question=question, target=target, response=response
        )
        try:
            raw = await self.client.complete(system=self.config.system_prompt, user=user_prompt)
            verdict = raw.strip().upper()
            if "CORRECT" == verdict:
                return True
            elif "INCORRECT" == verdict:
                return False
            else:
                raise AmbiguousVerdictError(f"Expected CORRECT or INCORRECT, got: {raw!r}")
        except (GradingError, asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as e:
            raise GradingError(f"Grading failed: {e}") from e
