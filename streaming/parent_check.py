"""Compare this run's parent headlines with the verified fraud-risk-mlops run.

Latency is left out of the comparison. The load-test percentile moves with the
machine. The model and drift headlines are deterministic for a fixed seed.
"""

from __future__ import annotations

import json
from pathlib import Path

REFERENCE_KEYS = (
    "n_transactions",
    "test_pr_auc",
    "pr_auc_lift",
    "recall_at_1pct",
    "week_11_max_psi",
)


def relative_error(current: float, reference: float) -> float:
    return abs(float(current) - float(reference)) / max(abs(float(reference)), 1e-12)


def max_relative_error(current: dict, reference: dict, keys: tuple[str, ...] = REFERENCE_KEYS) -> float:
    worst = 0.0
    for key in keys:
        worst = max(worst, relative_error(current[key], reference[key]))
    return float(worst)


def load_parent_reference(root: Path) -> dict:
    path = Path(root) / "streaming" / "parent_reference.json"
    return json.loads(path.read_text(encoding="utf-8"))
