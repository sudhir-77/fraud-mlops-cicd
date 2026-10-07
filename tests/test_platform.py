"""Model gate, alert rules, Dockerfile lint, and Kubernetes schema checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from streaming.dockerlint import lint_dockerfile, lint_dockerfile_text
from streaming.gate import allow_promotion, evaluate_production_and_degraded, gate_decision
from streaming.gate_check import main as gate_main
from streaming.manifests import manifests_ok, validate_document, validate_manifest_dir
from streaming.monitor import ScoringMonitor, alert_rate_rule, drift_rule, latency_rule


def test_gate_blocks_either_metric_dropping() -> None:
    assert gate_decision(0.80, 0.70, 0.80, 0.70)["passed"] is True
    assert gate_decision(0.90, 0.80, 0.80, 0.70)["passed"] is True
    dropped_pr = gate_decision(0.79, 0.70, 0.80, 0.70)
    dropped_recall = gate_decision(0.80, 0.69, 0.80, 0.70)
    assert dropped_pr["passed"] is False
    assert dropped_pr["pr_auc_below_production"] is True
    assert dropped_recall["passed"] is False
    assert dropped_recall["recall_at_1pct_below_production"] is True
    assert allow_promotion(dropped_pr) is False
    assert allow_promotion(gate_decision(0.80, 0.70, 0.80, 0.70)) is True


def test_degraded_scores_fail_on_a_known_split() -> None:
    y_true = np.array([0, 0, 0, 1, 0, 1, 0, 0], dtype=np.int8)
    scores = np.array([0.1, 0.2, 0.05, 0.9, 0.15, 0.8, 0.0, 0.3], dtype=np.float64)
    day = np.array(["1970-01-01"] * len(y_true))
    txn_id = np.arange(len(y_true))
    result = evaluate_production_and_degraded(y_true, scores, day, txn_id)
    assert result["production"]["passed"] is True
    assert result["degraded"]["passed"] is False
    assert result["degraded"]["pr_auc"] < result["production"]["pr_auc"]
    assert result["degraded"]["recall_at_1pct"] < result["production"]["recall_at_1pct"]
    assert gate_main() == 0


def test_alert_rules_match_hand_values() -> None:
    assert drift_rule(0.2) is False
    assert drift_rule(0.2000001) is True
    assert latency_rule(49.9) is False
    assert latency_rule(50.0) is True
    assert alert_rate_rule(0.01) is False
    assert alert_rate_rule(0.001) is False
    assert alert_rate_rule(0.0009) is True
    assert alert_rate_rule(0.05) is False
    assert alert_rate_rule(0.0501) is True
    monitor = ScoringMonitor()
    monitor.observe_batch(4, 0.004, 4, 0)
    monitor.set_week(11, 1.5, 1)
    monitor.set_alert_rate(0.01)
    text = monitor.render()
    assert "fraud_week_psi" in text
    assert "fraud_drift_alert" in text
    assert 'week="11"' in text


def test_dockerfile_lint_accepts_this_repo_and_rejects_latest() -> None:
    root = Path(__file__).resolve().parents[1]
    assert lint_dockerfile(root / "Dockerfile") == []
    problems = lint_dockerfile_text(
        "\n".join(
            [
                "FROM python:latest",
                "ADD app.py /app/app.py",
                "RUN sudo pip install app",
                "ENV API_KEY=secret",
                "USER root",
            ]
        )
    )
    assert any("latest" in item for item in problems)
    assert any("COPY" in item for item in problems)
    assert any("sudo" in item for item in problems)
    assert any("secret" in item for item in problems)
    assert any("USER" in item or "root" in item for item in problems)
    assert any("WORKDIR" in item for item in problems)
    assert any("CMD" in item for item in problems)


def test_manifests_validate_and_a_broken_deployment_does_not() -> None:
    root = Path(__file__).resolve().parents[1]
    report = validate_manifest_dir(root / "k8s")
    assert manifests_ok(report) is True
    assert {item["kind"] for item in report.values()} >= {
        "Deployment",
        "Service",
        "HorizontalPodAutoscaler",
    }
    valid, kind = validate_document(
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "broken"},
            "spec": "nope",
        }
    )
    assert kind == "Deployment"
    assert valid is False
