"""API reason codes, registry promotion, drift alerts, and report text."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api import create_app, create_app_from_env, top_reason_codes
from src.config import MODEL_FEATURES, PSI_ALERT, full_config, quick_config, train_config_for
from src.drift import alerts_from_psi, feature_psi_table
from src.plots import mean_abs_shap, save_drift_bars
from src.registry import promote_model
from src.report import finalize_metrics, write_resume_notes


def _tiny_booster() -> lgb.Booster:
    rng = np.random.default_rng(0)
    x_mat = np.zeros((200, 4), dtype=np.float64)
    x_mat[:, 0] = np.linspace(0, 1, 200)
    x_mat[:, 1] = rng.normal(size=200)
    y_true = (x_mat[:, 0] > 0.5).astype(np.int8)
    dataset = lgb.Dataset(x_mat, label=y_true, feature_name=["f0", "f1", "f2", "f3"])
    return lgb.train(
        {"objective": "binary", "verbosity": -1, "num_threads": 1, "seed": 0},
        dataset,
        num_boost_round=8,
    )


def test_reason_codes_pick_largest_abs_shap() -> None:
    contrib = np.array([0.1, -2.0, 0.4, 0.05])
    reasons = top_reason_codes(contrib, ["a", "b", "c"], k=2)
    assert [item["feature"] for item in reasons] == ["b", "c"]
    assert reasons[0]["direction"] == "decreases_risk"
    assert reasons[1]["direction"] == "increases_risk"


def test_score_endpoint_matches_booster_and_names_top_feature() -> None:
    booster = _tiny_booster()
    names = ["f0", "f1", "f2", "f3"]
    app = create_app(booster, threshold=0.5, feature_names=names)
    client = TestClient(app)
    features = {"f0": 0.9, "f1": 0.0, "f2": 0.0, "f3": 0.0}
    response = client.post("/score", json={"txn_id": 7, "features": features})
    assert response.status_code == 200
    body = response.json()
    row = np.array([[0.9, 0.0, 0.0, 0.0]])
    assert abs(body["score"] - float(booster.predict(row)[0])) < 1e-8
    assert body["reason_codes"][0]["feature"] == "f0"
    assert body["decision"] in {"review", "approve"}
    missing = client.post("/score", json={"txn_id": 1, "features": {"f0": 1.0}})
    assert missing.status_code == 400
    health = client.get("/health")
    assert health.status_code == 200


def test_create_app_from_env(tmp_path: Path, monkeypatch) -> None:
    booster = _tiny_booster()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    booster.save_model(str(model_dir / "fraud_lgbm.txt"))
    (model_dir / "serving.json").write_text(
        json.dumps(
            {
                "model_path": "models/fraud_lgbm.txt",
                "threshold": 0.5,
                "feature_names": ["f0", "f1", "f2", "f3"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FRAUD_PROJECT_ROOT", str(tmp_path))
    app = create_app_from_env()
    client = TestClient(app)
    response = client.post(
        "/score",
        json={"txn_id": 1, "features": {"f0": 0.2, "f1": 0.0, "f2": 0.0, "f3": 0.0}},
    )
    assert response.status_code == 200
    assert 0.0 <= response.json()["score"] <= 1.0


def test_registry_promotes_to_production(tmp_path: Path) -> None:
    booster = _tiny_booster()
    promoted = promote_model(
        booster,
        tmp_path / "mlruns",
        {"val_pr_auc": 0.5},
        {"num_trees": booster.num_trees()},
    )
    assert promoted["stage"] == "Production"
    assert promoted["version"]


def test_drift_alert_on_region_shift_only() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    rows = []
    for week, region in ((9, None), (10, None), (11, 0), (12, None)):
        if region is None:
            acq = np.arange(n) % 5
        else:
            acq = np.zeros(n, dtype=int)
        rows.append(
            pd.DataFrame(
                {
                    "week": np.full(n, week),
                    "acq_region": acq,
                    "amount": rng.normal(loc=40, scale=5, size=n),
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True)
    table = feature_psi_table(frame, 9, [10, 11, 12], ["acq_region", "amount"])
    alerts = alerts_from_psi(table)
    assert alerts[11] is True
    assert alerts[10] is False
    assert alerts[12] is False
    region_11 = table[(table["week"] == 11) & (table["feature"] == "acq_region")]["psi"].iloc[0]
    assert float(region_11) > PSI_ALERT


def test_peak_memory_sampler() -> None:
    from src.hardware import PeakMemory, host_info

    info = host_info()
    assert info["ram_gb"] > 0
    assert info["python"]
    sampler = PeakMemory(interval_s=0.05)
    sampler.start()
    assert sampler.stop() >= 0


def test_configs_and_shap_helper(tmp_path: Path) -> None:
    assert full_config().n_transactions == 3_000_000
    assert quick_config().n_per_week % 5 == 0
    assert train_config_for(False).num_boost_round > train_config_for(True).num_boost_round
    assert len(MODEL_FEATURES) == 15
    values = mean_abs_shap([np.array([[1.0, -3.0], [1.0, -1.0]]), np.zeros((2, 2))])
    assert values.shape == (2,)
    save_drift_bars([10, 11, 12], [0.01, 1.2, 0.02], 0.2, tmp_path / "drift.png")
    assert (tmp_path / "drift.png").exists()


def test_resume_notes_copy_json_numbers(tmp_path: Path) -> None:
    metrics = {
        "success_criteria": [],
        "metrics": {
            "headline": {
                "n_transactions": 3000000,
                "test_pr_auc": 0.8123,
                "pr_auc_lift": 2.15,
                "recall_at_1pct": 0.7812,
                "p95_ms": 12.5,
                "week_11_max_psi": 1.2345,
            },
            "drift": {"week_10_alert": 0, "week_11_alert": 1, "week_12_alert": 0},
        },
        "hardware": {},
        "runtime_seconds": 1.0,
    }
    metrics = finalize_metrics(
        metrics,
        {"cpu": "cpu", "ram_gb": 16, "python": "3.12.10", "os": "win"},
        1.2,
        3.5,
        88.0,
    )
    path = tmp_path / "RESUME_NOTES.md"
    write_resume_notes(path, metrics)
    text = path.read_text(encoding="utf-8")
    assert "3000000" in text
    assert "0.8123" in text
    assert "2.15" in text
    assert "12.5" in text
    assert "0.7812" in text
    assert "1.2345" in text
    assert "—" not in text
    assert metrics["hardware"]["peak_rss_gb"] == 1.2
    assert metrics["success_criteria"][-1]["name"] == "Test coverage of src/"
    assert metrics["success_criteria"][-1]["passed"] is True
