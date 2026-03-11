"""High-level judge abstractions: single-call and batched binary judging."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from tinker_cookbook.scoring.types import (
    BinaryJudgeClient,
    BinaryJudgeError,
)


class _JudgeTemplate(Protocol):
    """Duck-typed template protocol — accepts BinaryJudgeTemplate and JinjaBinaryJudgeTemplate."""

    def render(self, **kwargs: object) -> tuple[str, str]: ...
    def parse(self, response: str) -> bool: ...

K = TypeVar("K")


@dataclass(frozen=True)
class BatchResult(Generic[K]):
    values: dict[K, bool]
    errors: dict[K, BinaryJudgeError]


class AsyncBinaryJudge:
    def __init__(self, client: BinaryJudgeClient, template: _JudgeTemplate) -> None:
        self.client = client
        self.template = template

    async def judge(self, **kwargs: object) -> bool:
        system, user = self.template.render(**kwargs)
        response = await self.client.complete(system, user)
        return self.template.parse(response)


class JudgeBatch(Generic[K]):
    def __init__(self, judge: AsyncBinaryJudge) -> None:
        self._judge = judge
        self._pending: dict[K, dict[str, object]] = {}
        self._resolved: dict[K, bool] = {}
        self._ran = False

    def add(self, key: K, **kwargs: object) -> None:
        if self._ran:
            raise RuntimeError("Batch already executed")
        if key in self._resolved:
            return  # already resolved, skip
        if key in self._pending:
            if self._pending[key] != kwargs:
                raise ValueError(
                    f"Key {key!r} already pending with different kwargs: "
                    f"existing={self._pending[key]!r}, new={kwargs!r}"
                )
            return  # dedupe
        self._pending[key] = kwargs

    def resolve(self, key: K, value: bool) -> None:
        if self._ran:
            raise RuntimeError("Batch already executed")
        if key in self._pending:
            raise ValueError(
                f"Key {key!r} already pending with kwargs={self._pending[key]!r}; "
                f"cannot resolve"
            )
        if key in self._resolved:
            if self._resolved[key] != value:
                raise ValueError(
                    f"Key {key!r} already resolved to {self._resolved[key]!r}, "
                    f"cannot change to {value!r}"
                )
            return  # dedupe
        self._resolved[key] = value

    async def run(self) -> BatchResult[K]:
        if self._ran:
            raise RuntimeError("Batch already executed")
        self._ran = True

        if not self._pending:
            return BatchResult(values=dict(self._resolved), errors={})

        keys = list(self._pending.keys())
        coros = [self._judge.judge(**self._pending[k]) for k in keys]
        results = await asyncio.gather(*coros, return_exceptions=True)

        values = dict(self._resolved)
        errors: dict[K, BinaryJudgeError] = {}

        for key, result in zip(keys, results):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, BaseException):
                if isinstance(result, BinaryJudgeError):
                    errors[key] = result
                else:
                    errors[key] = BinaryJudgeError(str(result))
                continue
            values[key] = result

        return BatchResult(values=values, errors=errors)
