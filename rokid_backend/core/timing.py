"""Per-stage timing used to report where the latency of an answer goes."""
from __future__ import annotations

import time
from contextlib import contextmanager


class StageTimer:
    def __init__(self) -> None:
        self._started = time.perf_counter()
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round(self.stages.get(name, 0.0) + (time.perf_counter() - start) * 1000, 2)

    def mark(self, name: str, value_ms: float = 0.0) -> None:
        self.stages.setdefault(name, round(value_ms, 2))

    @property
    def total_ms(self) -> int:
        return int((time.perf_counter() - self._started) * 1000)
