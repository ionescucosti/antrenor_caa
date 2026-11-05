"""Idempotent replies per request_id. The Rokid client retries the same
request_id up to three times on 502/503/504 or network errors, so the first
computed reply is stored for a while and returned again unchanged; a retry that
arrives while the first attempt is still running waits for it instead of
starting a second LLM call."""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Any


class IdempotencyCache:
    def __init__(self, ttl_seconds: int = 600, max_entries: int = 2000) -> None:
        self.ttl = max(0, ttl_seconds)
        self.max_entries = max_entries
        self._entries: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}

    def get(self, key: str):
        item = self._entries.get(key)
        if item is None:
            return None
        expires, value = item
        if expires < time.monotonic():
            self._entries.pop(key, None)
            return None
        self._entries.move_to_end(key)
        return value

    def put(self, key: str, value: Any) -> None:
        if self.ttl == 0:
            return
        self._entries[key] = (time.monotonic() + self.ttl, value)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def lock(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            if len(self._locks) > 5000:
                self._locks = {k: v for k, v in self._locks.items() if v.locked()}
            lock = self._locks[key] = asyncio.Lock()
        return lock

    def release(self, key: str) -> None:
        lock = self._locks.get(key)
        if lock is not None and not lock.locked():
            self._locks.pop(key, None)
