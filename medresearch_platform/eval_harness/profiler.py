"""CPU/RAM/time/size resource profiler for training and inference."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil


@dataclass
class ResourceMetrics:
    wall_seconds: float
    peak_rss_mb: float
    cpu_percent: float
    docs_count: int = 0
    ms_per_doc: float = 0.0
    docs_per_second: float = 0.0
    model_size_bytes: int = 0
    parameter_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResourceProfiler:
    """Context manager measuring wall-clock duration, peak RSS MB, and CPU% via psutil."""

    def __init__(self, *, docs_count: int = 0, parameter_count: int = 0, model_path: Path | None = None) -> None:
        self.docs_count = docs_count
        self.parameter_count = parameter_count
        self.model_path = model_path
        self._proc = psutil.Process()
        self._t0 = 0.0
        self._rss_before = 0.0
        self.metrics: ResourceMetrics | None = None

    def __enter__(self) -> ResourceProfiler:
        self._proc.cpu_percent(interval=None)
        self._rss_before = self._proc.memory_info().rss / (1024 * 1024)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        elapsed = max(1e-6, time.perf_counter() - self._t0)
        rss_after = self._proc.memory_info().rss / (1024 * 1024)
        cpu_pct = self._proc.cpu_percent(interval=None)
        model_bytes = self.model_path.stat().st_size if (self.model_path and self.model_path.exists()) else 0
        ms_per_doc = (elapsed * 1000.0 / self.docs_count) if self.docs_count > 0 else 0.0
        docs_per_s = (self.docs_count / elapsed) if self.docs_count > 0 else 0.0
        self.metrics = ResourceMetrics(
            wall_seconds=round(elapsed, 6),
            peak_rss_mb=round(max(self._rss_before, rss_after), 3),
            cpu_percent=round(cpu_pct, 2),
            docs_count=self.docs_count,
            ms_per_doc=round(ms_per_doc, 4),
            docs_per_second=round(docs_per_s, 3),
            model_size_bytes=model_bytes,
            parameter_count=self.parameter_count,
        )
