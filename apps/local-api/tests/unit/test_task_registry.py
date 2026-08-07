import asyncio

from core.tasks import BackgroundTaskRegistry


async def test_spawn_runs_the_coroutine_to_completion():
    registry = BackgroundTaskRegistry()
    result = {}

    async def _work():
        result["ran"] = True

    task = registry.spawn(_work(), name="test")
    await task

    assert result == {"ran": True}


async def test_drain_waits_for_in_flight_tasks():
    registry = BackgroundTaskRegistry()
    result = {}

    async def _work():
        await asyncio.sleep(0.01)
        result["ran"] = True

    registry.spawn(_work(), name="test")
    await registry.drain(timeout=1.0)

    assert result == {"ran": True}


async def test_drain_cancels_tasks_that_exceed_the_timeout():
    registry = BackgroundTaskRegistry()
    started = asyncio.Event()
    cancelled = False

    async def _hangs_forever():
        nonlocal cancelled
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled = True
            raise

    registry.spawn(_hangs_forever(), name="test")
    await started.wait()
    await registry.drain(timeout=0.05)

    assert cancelled is True


async def test_drain_with_no_tasks_returns_immediately():
    registry = BackgroundTaskRegistry()
    await registry.drain(timeout=1.0)  # must not hang or raise
