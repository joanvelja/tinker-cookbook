from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


async def run_with_heartbeat(
    awaitable: Awaitable[T],
    *,
    label: str,
    interval_s: float = 30.0,
) -> T:
    start = time.monotonic()
    print(f"[start] {label}", flush=True)

    if interval_s <= 0:
        result = await awaitable
        elapsed = time.monotonic() - start
        print(f"[done] {label} ({elapsed:.1f}s)", flush=True)
        return result

    done = asyncio.Event()

    async def _heartbeat() -> None:
        while True:
            try:
                await asyncio.wait_for(done.wait(), timeout=interval_s)
                return
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - start
                print(f"[heartbeat] {label} still running ({elapsed:.0f}s)", flush=True)

    heartbeat_task = asyncio.create_task(_heartbeat())
    try:
        result = await awaitable
    finally:
        done.set()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task

    elapsed = time.monotonic() - start
    print(f"[done] {label} ({elapsed:.1f}s)", flush=True)
    return result
