"""End-to-end quick run. Thresholds are enforced on the full `python run.py`, not here."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import train_config_for
from src.metrics import pr_auc
from src.pipeline import run_pipeline


def test_full_load_test_covers_at_least_one_minute() -> None:
    cfg = train_config_for(False)
    assert cfg.load_rps == 50
    assert cfg.load_seconds >= 60
    assert cfg.load_rps * cfg.load_seconds >= 3000


def test_quick_pipeline(tmp_path: Path) -> None:
    metrics = run_pipeline(tmp_path, quick=True)
    predictions = pd.read_csv(tmp_path / "results" / "test_predictions.csv")
    assert set(predictions["week"].unique()) <= {10, 11, 12}
    assert "txn_id" in predictions.columns
    recomputed = pr_auc(predictions["y_true"], predictions["model_score"])
    assert abs(recomputed - metrics["metrics"]["model"]["test_pr_auc"]) < 1e-9
    splits = metrics["splits"]["transactions"]
    assert splits["train"] > 0 and splits["val"] > 0 and splits["test"] > 0
    assert splits["train"] + splits["val"] + splits["test"] == metrics["metrics"]["headline"]["n_transactions"]
    drift = pd.read_csv(tmp_path / "results" / "drift_weekly.csv")

    def psi(week: int, feature: str) -> float:
        match = drift[(drift["week"] == week) & (drift["feature"] == feature)]
        return float(match["psi"].iloc[0])

    assert psi(11, "acq_region") > 0.2
    assert psi(10, "acq_region") < 0.05
    assert psi(12, "acq_region") < 0.05
    assert metrics["metrics"]["data"]["pit_future_feature_violations"] == 0
    assert (tmp_path / "results" / "pr_curve.png").exists()
    assert (tmp_path / "results" / "drift_weekly.md").exists()
    assert metrics["metrics"]["serving"]["mlflow_stage"] == "Production"
    assert metrics["metrics"]["serving"]["n_requests"] == 50
