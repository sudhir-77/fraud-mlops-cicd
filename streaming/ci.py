"""Lint, tests, and the model gate. GitHub Actions and `python run.py` both call this."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def commands(root: Path) -> list[list[str]]:
    root = Path(root)
    coverage = root / "results" / "coverage.json"
    return [
        [sys.executable, "-m", "ruff", "check", "src", "streaming", "tests", "run.py"],
        [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "-q",
            "--cov=src",
            "--cov=streaming",
            "--cov-report=term-missing",
            f"--cov-report=json:{coverage}",
            "--cov-fail-under=75",
        ],
        [sys.executable, "-m", "streaming.gate_check"],
    ]


def run_commands(root: Path) -> int:
    root = Path(root)
    for command in commands(root):
        print("ci step " + " ".join(command), flush=True)
        status = subprocess.run(command, cwd=root, check=False).returncode
        if status != 0:
            return int(status)
    return 0


def split_coverage(path: Path) -> tuple[float, float]:
    """Return (src percent, src+streaming percent), weighted by statement count."""
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    src_weight = src_count = all_weight = all_count = 0.0
    for name, info in payload.get("files", {}).items():
        summary = info.get("summary", {})
        count = float(summary.get("num_statements", 0))
        percent = float(summary.get("percent_covered", 0.0))
        all_weight += percent * count
        all_count += count
        norm = str(name).replace("\\", "/")
        if "/streaming/" in f"/{norm}":
            continue
        if norm.startswith("src/") or "/src/" in f"/{norm}":
            src_weight += percent * count
            src_count += count
    if all_count <= 0 or src_count <= 0:
        raise RuntimeError(f"coverage report {path} has no statements")
    return src_weight / src_count, all_weight / all_count


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    return run_commands(root)


if __name__ == "__main__":
    raise SystemExit(main())
