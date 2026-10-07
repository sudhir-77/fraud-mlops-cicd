"""Hand-written rules on amount, short-window velocity, and new-device plus high amount.

Thresholds are chosen on training rows only, by average precision. The score is a
fixed weighted sum so the search cannot invent new signals such as merchant risk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score

AMOUNT_W = 0.40
VEL_W = 0.45
NEW_W = 0.70
TIE_W = 0.01


@dataclass(frozen=True)
class RuleParams:
    amount_threshold: float
    velocity_threshold: float
    new_device_amount_threshold: float

    def as_dict(self) -> dict[str, float]:
        return {
            "amount_threshold": self.amount_threshold,
            "velocity_threshold": self.velocity_threshold,
            "new_device_amount_threshold": self.new_device_amount_threshold,
        }


def rule_scores(
    amount: np.ndarray,
    velocity_1h: np.ndarray,
    is_new_device: np.ndarray,
    params: RuleParams,
) -> np.ndarray:
    amount = np.asarray(amount, dtype=np.float64)
    velocity_1h = np.asarray(velocity_1h, dtype=np.float64)
    is_new_device = np.asarray(is_new_device, dtype=np.float64)
    high_amount = (amount >= params.amount_threshold).astype(np.float64)
    high_velocity = (velocity_1h >= params.velocity_threshold).astype(np.float64)
    new_and_high = (
        (is_new_device >= 0.5) & (amount >= params.new_device_amount_threshold)
    ).astype(np.float64)
    # Tiny amount term breaks ties inside a rule bucket without reordering buckets.
    tie = np.log1p(np.maximum(amount, 0.0)) / 20.0
    return AMOUNT_W * high_amount + VEL_W * high_velocity + NEW_W * new_and_high + TIE_W * tie


def tune_rules(
    amount: np.ndarray,
    velocity_1h: np.ndarray,
    is_new_device: np.ndarray,
    y_true: np.ndarray,
) -> tuple[RuleParams, float]:
    """Grid search on the provided rows. Callers must pass training weeks only."""
    amount = np.asarray(amount, dtype=np.float64)
    y_true = np.asarray(y_true)
    amount_qs = (0.90, 0.95, 0.97, 0.99)
    velocity_grid = (2.0, 3.0, 4.0, 5.0, 6.0)
    new_qs = (0.80, 0.90, 0.95)
    quantiles = {q: float(np.quantile(amount, q)) for q in set(amount_qs + new_qs)}
    best: RuleParams | None = None
    best_ap = -1.0
    for amount_q in amount_qs:
        for velocity in velocity_grid:
            for new_q in new_qs:
                params = RuleParams(quantiles[amount_q], float(velocity), quantiles[new_q])
                scores = rule_scores(amount, velocity_1h, is_new_device, params)
                ap = float(average_precision_score(y_true, scores))
                if ap > best_ap:
                    best_ap = ap
                    best = params
    if best is None:
        raise RuntimeError("rule search did not evaluate a candidate")
    return best, best_ap
