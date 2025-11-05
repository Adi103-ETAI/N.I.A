import asyncio
import pytest
import time

from core.tool_manager import ToolManager


def test_sync_tool_execution():
    mgr = ToolManager()

    def add(a: int = 0, b: int = 0):
        return a + b

    mgr.register('add', add)
    result = mgr.execute('add', {'a': 2, 'b': 3})
    assert result == 5


def test_async_tool_execution():
    mgr = ToolManager()

    async def async_add(a: int = 0, b: int = 0):
        return a + b

    mgr.register('async_add', async_add)
    result = asyncio.run(mgr.execute_async('async_add', {'a': 4, 'b': 1}))
    assert result == 5


def test_sync_timeout_raises_runtime_error():
    mgr = ToolManager()

    def slow():
        time.sleep(1.0)
        return 'done'

    mgr.register('slow', slow)

    with pytest.raises(RuntimeError) as exc:
        mgr.execute('slow', {}, timeout=0.1)

    assert 'timeout' in str(exc.value).lower() or 'exceeded' in str(exc.value).lower()


def test_async_timeout_raises_runtime_error():
    mgr = ToolManager()

    async def slow_async():
        await asyncio.sleep(1.0)
        return 'done'

    mgr.register('slow_async', slow_async)

    with pytest.raises(RuntimeError):
        asyncio.run(mgr.execute_async('slow_async', {}, timeout=0.1))
