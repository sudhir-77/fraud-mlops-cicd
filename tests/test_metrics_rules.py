"""Hand-computed checks for ranking metrics, PSI, cost, and the rules score."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score

from src.metrics import (
    chi2_homogeneity,
    evaluate_op,
    expected_cost,
    pr_auc,
    psi_from_counts,
    recall_at_alert_rate,
    select_threshold,
)
from src.rules import RuleParams, rule_scores


def test_pr_auc_matches_sklearn() -> None:
    y_true = np.array([0, 1, 1, 0])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    assert pr_auc(y_true, scores) == float(average_precision_score(y_true, scores))


def test_recall_at_alert_rate_hand_example() -> None:
    day = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    y_true = np.array([1, 0, 0, 0, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.05, 0.2, 0.95, 0.1, 0.05])
    txn_id = np.arange(8)
    assert recall_at_alert_rate(y_true, scores, day, txn_id, rate=0.25) == 1.0
    scores_miss = np.array([0.1, 0.9, 0.2, 0.05, 0.9, 0.1, 0.2, 0.05])
    assert recall_at_alert_rate(y_true, scores_miss, day, txn_id, rate=0.25) == 0.0


def test_recall_tie_breaks_on_smaller_txn_id() -> None:
    day = np.array([1, 1])
    y_true = np.array([0, 1])
    scores = np.array([0.5, 0.5])
    # k = floor(2 * 0.5) = 1. Smaller txn_id wins the tie and is legitimate, so recall is 0.
    caught = recall_at_alert_rate(y_true, scores, day, np.array([1, 2]), rate=0.5)
    assert caught == 0.0
    caught_other = recall_at_alert_rate(y_true, scores, day, np.array([2, 1]), rate=0.5)
    assert caught_other == 1.0


def test_psi_hand_value() -> None:
    # (0.6-0.1)*ln(6) + (0.1-0.6)*ln(1/6) = ln(6)
    psi = psi_from_counts(np.array([10, 30, 60]), np.array([60, 30, 10]))
    assert math.isclose(psi, math.log(6.0), rel_tol=1e-12)


def test_chi2_hand_value() -> None:
    stat = chi2_homogeneity(np.array([10.0, 0.0]), np.array([0.0, 10.0]))
    assert math.isclose(stat, 20.0, rel_tol=1e-12)


def test_threshold_finds_zero_cost() -> None:
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    threshold, cost = select_threshold(y_true, scores, cost_fp=1.0, cost_fn=1.0)
    assert cost == 0.0
    assert expected_cost(y_true, scores, threshold, 1.0, 1.0) == 0.0
    assert 0.4 < threshold <= 0.6


def test_evaluate_op() -> None:
    assert evaluate_op(0.2, ">=", 0.2)
    assert not evaluate_op(0.2, ">", 0.2)
    assert evaluate_op(3, "<", 4)
    assert evaluate_op(3, "<=", 3)
    assert evaluate_op(1.5, "between", 1.0, 2.0)
    assert not evaluate_op(2.5, "between", 1.0, 2.0)


def test_rule_scores_rank_ato_above_amount_only() -> None:
    params = RuleParams(amount_threshold=100.0, velocity_threshold=3.0, new_device_amount_threshold=80.0)
    amount = np.array([40.0, 500.0, 800.0, 4.0])
    velocity = np.array([0.0, 0.0, 0.0, 6.0])
    is_new = np.array([0.0, 0.0, 1.0, 0.0])
    scores = rule_scores(amount, velocity, is_new, params)
    # Bill (high amount, known device) loses to takeover (high amount, new device)
    # and to a velocity burst.
    assert scores[2] > scores[1]
    assert scores[3] > scores[0]
    assert scores[0] < scores[1]
