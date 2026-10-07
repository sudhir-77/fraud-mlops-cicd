"""CI step: the production scores pass the gate and the inverted scores do not.

This does not load the trained model. `python run.py` writes the real
validation-week gate file; this module is the same rule on a fixed sample.
"""

from __future__ import annotations

import numpy as np

from streaming.gate import allow_promotion, evaluate_production_and_degraded


def synthetic_labels() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.zeros(400, dtype=np.int8)
    y_true[:40] = 1
    scores = np.where(y_true == 1, 0.9, 0.1).astype(np.float64)
    day = np.array(["1970-01-01"] * y_true.shape[0])
    txn_id = np.arange(y_true.shape[0], dtype=np.int64)
    return y_true, scores, day, txn_id


def main() -> int:
    y_true, scores, day, txn_id = synthetic_labels()
    result = evaluate_production_and_degraded(y_true, scores, day, txn_id)
    if not allow_promotion(result["production"]):
        print("synthetic production candidate was blocked", flush=True)
        return 1
    if allow_promotion(result["degraded"]):
        print("synthetic degraded candidate was promoted", flush=True)
        return 1
    print(
        "model gate passed production "
        f"(PR-AUC {result['production']['pr_auc']:.4f}) and blocked degraded "
        f"(PR-AUC {result['degraded']['pr_auc']:.4f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
