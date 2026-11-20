"""
Clima-Core v1.2 Benchmark Utilities.
Tracks peak RAM (MB), wall-clock duration (seconds), thread counts, and model sizes.
"""

from __future__ import annotations

import gc
import os
import time
import psutil
from pathlib import Path
from typing import Dict, Any, Callable


class ResourceTracker:
    def __init__(self, task_name: str, max_threads: int = 2):
        self.task_name = task_name
        self.max_threads = max_threads
        self.process = psutil.Process(os.getpid())
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.start_ram_mb: float = 0.0
        self.peak_ram_mb: float = 0.0

    def __enter__(self):
        # Configure thread limits across libraries
        os.environ["OMP_NUM_THREADS"] = str(self.max_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.max_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(self.max_threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(self.max_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(self.max_threads)

        gc.collect()
        self.start_ram_mb = self.process.memory_info().rss / (1024 * 1024)
        self.peak_ram_mb = self.start_ram_mb
        self.start_time = time.perf_counter()
        return self

    def sample_ram(self):
        current_ram = self.process.memory_info().rss / (1024 * 1024)
        if current_ram > self.peak_ram_mb:
            self.peak_ram_mb = current_ram

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        gc.collect()
        current_ram = self.process.memory_info().rss / (1024 * 1024)
        if current_ram > self.peak_ram_mb:
            self.peak_ram_mb = current_ram

    @property
    def duration_seconds(self) -> float:
        return round(self.end_time - self.start_time, 3)

    @property
    def peak_ram_delta_mb(self) -> float:
        return round(self.peak_ram_mb - self.start_ram_mb, 2)

    @property
    def max_ram_observed_mb(self) -> float:
        return round(self.peak_ram_mb, 2)

    def summary(self, artifact_path: Path | None = None) -> Dict[str, Any]:
        artifact_kb = 0.0
        if artifact_path and Path(artifact_path).exists():
            artifact_kb = round(Path(artifact_path).stat().st_size / 1024, 2)

        return {
            "task_name": self.task_name,
            "duration_seconds": self.duration_seconds,
            "start_ram_mb": round(self.start_ram_mb, 2),
            "peak_ram_mb": self.max_ram_observed_mb,
            "net_ram_increase_mb": self.peak_ram_delta_mb,
            "max_cpu_threads": self.max_threads,
            "artifact_size_kb": artifact_kb,
        }
