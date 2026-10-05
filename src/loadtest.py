"""Local load test: 50 requests per second against a loopback uvicorn worker."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import psutil


def summarize_latencies(latencies_ms: np.ndarray) -> dict[str, float]:
    values = np.asarray(latencies_ms, dtype=np.float64)
    if values.size == 0:
        raise ValueError("no latencies")
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "n_requests": float(values.size),
        "max_ms": float(values.max()),
    }


async def _collect(url: str, payloads: list[dict], rps: int, seconds: int) -> np.ndarray:
    timeout = httpx.Timeout(5.0)
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=50)
    n = rps * seconds
    interval = 1.0 / rps
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for _ in range(min(40, max(10, rps))):
            response = await client.post(url, json=payloads[0])
            response.raise_for_status()

        async def one(i: int) -> float:
            target = start + i * interval
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            payload = payloads[i % len(payloads)]
            t0 = time.perf_counter()
            response = await client.post(url, json=payload)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            response.raise_for_status()
            return elapsed_ms

        start = time.perf_counter()
        latencies = await asyncio.gather(*[one(i) for i in range(n)])
    return np.asarray(latencies, dtype=np.float64)


def run_load_test(
    base_url: str,
    payloads: list[dict],
    rps: int = 50,
    seconds: int = 4,
) -> np.ndarray:
    if rps < 1 or seconds < 1:
        raise ValueError("rps and seconds must be positive")
    if not payloads:
        raise ValueError("payloads required")
    return asyncio.run(_collect(base_url.rstrip("/") + "/score", payloads, rps, seconds))


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
    except psutil.Error:
        children = []
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=8)
    for child in children:
        try:
            child.kill()
        except psutil.Error:
            pass


def _wait_until_healthy(url: str, proc: subprocess.Popen, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    last_error = "server did not become healthy"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"scoring server exited early with code {proc.returncode}")
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code == 200:
                return
            last_error = f"status {response.status_code}"
        except httpx.HTTPError as exc:
            last_error = str(exc)
        time.sleep(0.2)
    raise TimeoutError(last_error)


def serve_and_measure(
    root: Path,
    payloads: list[dict],
    rps: int,
    seconds: int,
) -> np.ndarray:
    """Start uvicorn, measure, and always stop the child process."""
    port = _free_port()
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "uvicorn.log"
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["FRAUD_PROJECT_ROOT"] = str(root)
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api:create_app_from_env",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
        "--no-access-log",
        "--workers",
        "1",
    ]
    log_handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    try:
        health = f"http://127.0.0.1:{port}/health"
        _wait_until_healthy(health, proc)
        return run_load_test(f"http://127.0.0.1:{port}", payloads, rps=rps, seconds=seconds)
    finally:
        _stop_process(proc)
        log_handle.close()
