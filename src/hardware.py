"""Host description and a same-process peak RSS sampler, including child processes."""

from __future__ import annotations

import platform
import threading
import time

import psutil


def host_info() -> dict:
    memory = psutil.virtual_memory()
    cpu = platform.processor() or platform.machine()
    return {
        "cpu": cpu,
        "ram_gb": round(memory.total / (1024**3), 2),
        "gpu": None,
        "python": platform.python_version(),
        "os": platform.platform(),
    }


class PeakMemory:
    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="peak-rss", daemon=True)

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join(timeout=2)
        self._sample()
        return self.peak_gb

    @property
    def peak_gb(self) -> float:
        return round(self.peak_bytes / (1024**3), 4)

    def _sample(self) -> None:
        process = psutil.Process()
        total = int(process.memory_info().rss)
        try:
            children = process.children(recursive=True)
        except psutil.Error:
            children = []
        for child in children:
            try:
                total += int(child.memory_info().rss)
            except psutil.Error:
                continue
        if total > self.peak_bytes:
            self.peak_bytes = total

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            time.sleep(self.interval_s)
