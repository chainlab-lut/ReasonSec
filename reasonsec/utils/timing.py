from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class Timer:
    def __init__(self) -> None:
        self._start: float | None = None
        self.elapsed_seconds: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._start is not None:
            self.elapsed_seconds = time.perf_counter() - self._start
            self._start = None

    @property
    def elapsed_minutes(self) -> float:
        return self.elapsed_seconds / 60.0

    @property
    def elapsed_milliseconds(self) -> float:
        return self.elapsed_seconds * 1000.0


class StageTimer:
    def __init__(self) -> None:
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[Timer]:
        timer = Timer()
        with timer:
            yield timer
        self.stages[name] = self.stages.get(name, 0.0) + timer.elapsed_seconds

    def record(self, name: str, seconds: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + seconds

    def as_seconds(self) -> dict[str, float]:
        return dict(self.stages)

    def as_minutes(self) -> dict[str, float]:
        return {name: seconds / 60.0 for name, seconds in self.stages.items()}

    @property
    def total_seconds(self) -> float:
        return float(sum(self.stages.values()))
