"""Reproduce every result: python run.py, or python run.py --quick for a small run."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# Cap threads before NumPy loads via src.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"


def _coverage_percent(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["totals"]["percent_covered"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate the simulated fraud scorer.")
    parser.add_argument("--quick", action="store_true", help="Smaller run for tests, under a few minutes.")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Accepted for compatibility. Training stays on CPU.",
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    os.chdir(root)

    from src.hardware import PeakMemory, host_info
    from src.pipeline import run_pipeline
    from src.report import finalize_metrics, update_readme, write_resume_notes

    if args.device == "cuda":
        print("CUDA was requested; this project trains and serves on CPU.", flush=True)

    peak = PeakMemory()
    peak.start()
    started = time.perf_counter()
    metrics = run_pipeline(root, quick=args.quick, device=args.device)
    coverage = None
    if not args.quick:
        gc.collect()
        import pytest

        cov_path = root / "results" / "coverage.json"
        status = pytest.main(
            [
                str(root / "tests"),
                "-q",
                "--cov=src",
                "--cov-report=term-missing",
                f"--cov-report=json:{cov_path}",
            ]
        )
        if not cov_path.exists():
            raise RuntimeError(f"pytest did not write coverage (exit {status})")
        coverage = _coverage_percent(cov_path)
        metrics["metrics"]["pytest_exit_code"] = int(status)
    host = host_info()
    metrics = finalize_metrics(
        metrics,
        host,
        peak.stop(),
        round(time.perf_counter() - started, 3),
        coverage,
    )
    results = root / "results" / "metrics.json"
    results.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    update_readme(root / "README.md", metrics)
    write_resume_notes(root / "RESUME_NOTES.md", metrics)
    print(f"runtime_seconds {metrics['runtime_seconds']}", flush=True)
    print(f"peak_rss_gb {metrics['hardware']['peak_rss_gb']}", flush=True)
    failed = False
    for item in metrics["success_criteria"]:
        state = "PASS" if item["passed"] else "FAIL"
        if not item["passed"]:
            failed = True
        print(f"{state} {item['name']}: {item['value']} {item['op']} {item['threshold']}", flush=True)
    if args.quick:
        return 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
