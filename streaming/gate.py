"""Block promotion when PR-AUC or recall at 1% falls below production."""

from __future__ import annotations

import numpy as np

from src.config import ALERT_RATE
from src.metrics import pr_auc, recall_at_alert_rate


def gate_decision(
    pr_auc_value: float,
    recall_at_1pct: float,
    production_pr_auc: float,
    production_recall_at_1pct: float,
) -> dict:
    pr_value = float(pr_auc_value)
    recall_value = float(recall_at_1pct)
    prod_pr = float(production_pr_auc)
    prod_recall = float(production_recall_at_1pct)
    pr_below = pr_value < prod_pr
    recall_below = recall_value < prod_recall
    return {
        "passed": (not pr_below) and (not recall_below),
        "pr_auc": pr_value,
        "recall_at_1pct": recall_value,
        "production_pr_auc": prod_pr,
        "production_recall_at_1pct": prod_recall,
        "pr_auc_below_production": pr_below,
        "recall_at_1pct_below_production": recall_below,
    }


def allow_promotion(decision: dict) -> bool:
    return bool(decision["passed"])


def degraded_scores(y_true: np.ndarray) -> np.ndarray:
    """Invert the label so fraud ranks below legitimate traffic."""
    y_true = np.asarray(y_true)
    return np.where(y_true == 1, 0.0, 1.0).astype(np.float64)


def evaluate_production_and_degraded(
    y_true: np.ndarray,
    scores: np.ndarray,
    day: np.ndarray,
    txn_id: np.ndarray,
) -> dict:
    """Score the production vector and a deliberately worse vector on the same rows.

    The rows must be the validation week. The test weeks are not used here.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    day = np.asarray(day).astype(str)
    txn_id = np.asarray(txn_id)
    prod_pr = pr_auc(y_true, scores)
    prod_recall = recall_at_alert_rate(y_true, scores, day, txn_id, ALERT_RATE)
    production = gate_decision(prod_pr, prod_recall, prod_pr, prod_recall)
    worse = degraded_scores(y_true)
    degraded = gate_decision(
        pr_auc(y_true, worse),
        recall_at_alert_rate(y_true, worse, day, txn_id, ALERT_RATE),
        prod_pr,
        prod_recall,
    )
    return {"production": production, "degraded": degraded}
