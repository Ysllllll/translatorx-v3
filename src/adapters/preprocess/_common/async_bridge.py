"""Sync → async event-loop bridge for adapter backends.

The registry Backend contract is synchronous (``list[str] -> ...``) for
simplicity, but the LLM backends execute ``asyncio.gather`` internally.
Running ``asyncio.run`` directly fails when the caller is already inside
a running loop (tests, notebooks, FastAPI request handlers), so we
detect that case and relay the coroutine through a short-lived worker
thread.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


def run_async_in_sync(coro_factory: Callable[[], Awaitable[T]]) -> T:
    """Run ``coro_factory()`` to completion from synchronous code.

    * No running loop → plain :func:`asyncio.run`.
    * Running loop present (e.g. inside an ``async def`` test) → delegate
      to a single-shot worker thread so the running loop stays untouched.

    The *coro_factory* is a zero-arg callable rather than a coroutine so
    the worker thread can construct a fresh coroutine inside its own
    event loop (coroutines can only run once and can't cross loops).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro_factory())).result()
    return asyncio.run(coro_factory())


__all__ = ["run_async_in_sync"]
