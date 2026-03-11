from __future__ import annotations

import asyncio

import pytest

from tinker_cookbook.recipes.multiplayer_rl.debate.progress import run_with_heartbeat


@pytest.mark.asyncio
async def test_run_with_heartbeat_prints_start_and_done(capsys) -> None:
    result = await run_with_heartbeat(asyncio.sleep(0, result="ok"), label="quick op", interval_s=60)

    out = capsys.readouterr().out
    assert result == "ok"
    assert "[start] quick op" in out
    assert "[done] quick op" in out
    assert "[heartbeat]" not in out


@pytest.mark.asyncio
async def test_run_with_heartbeat_emits_periodic_heartbeat(capsys) -> None:
    result = await run_with_heartbeat(
        asyncio.sleep(0.03, result="slow"),
        label="slow op",
        interval_s=0.01,
    )

    out = capsys.readouterr().out
    assert result == "slow"
    assert "[start] slow op" in out
    assert "[heartbeat] slow op still running" in out
    assert "[done] slow op" in out
