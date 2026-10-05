"""Metric functions with deterministic tie breaks.

Recall at a daily alert rate uses floor(n * rate) rows per calendar day, at least 1
when the day is non-empty. Ties break toward the smaller txn_id.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        raise ValueError("PR-AUC needs both classes")
    return float(average_precision_score(y_true, scores))


def recall_at_alert_rate(
    y_true: np.ndarray,
    scores: np.ndarray,
    day: np.ndarray,
    txn_id: np.ndarray,
    rate: float = 0.01,
) -> float:
    """Fraction of fraud rows that fall in each day's top `rate` scores."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    day = np.asarray(day)
    txn_id = np.asarray(txn_id)
    total = float(y_true.sum())
    if total <= 0:
        return 0.0
    caught = 0.0
    for d in np.unique(day):
        mask = day == d
        yd = y_true[mask]
        sd = scores[mask]
        ids = txn_id[mask]
        n = int(yd.shape[0])
        k = int(math.floor(n * rate))
        if k < 1:
            k = 1 if n > 0 else 0
        if k == 0:
            continue
        # lexsort: last key is primary. Higher score first, then smaller txn_id.
        order = np.lexsort((ids, -sd))
        caught += float(yd[order[:k]].sum())
    return float(caught / total)


def expected_cost(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    cost_fp: float,
    cost_fn: float,
) -> float:
    y_true = np.asarray(y_true).astype(np.int8, copy=False)
    pred = np.asarray(scores) >= threshold
    fp = int(np.count_nonzero(pred & (y_true == 0)))
    fn = int(np.count_nonzero(~pred & (y_true == 1)))
    return float(cost_fp * fp + cost_fn * fn)


def select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    cost_fp: float,
    cost_fn: float,
) -> tuple[float, float]:
    """Lowest expected cost on a fixed probability grid.

    Candidates are k/5000 and do not depend on the score values, so a small
    change in a quantile cannot move the reported threshold. When several grid
    points share that minimum cost, the midpoint of the tied run is returned.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true).astype(np.int8, copy=False)
    if scores.size == 0:
        raise ValueError("no scores")
    n_steps = 5000
    candidates = np.arange(n_steps + 1, dtype=np.float64) / float(n_steps)
    y_pos = y_true == 1
    y_neg = ~y_pos
    n_pos = int(np.count_nonzero(y_pos))
    costs = np.empty(candidates.shape[0], dtype=np.float64)
    chunk = 100
    for start in range(0, candidates.shape[0], chunk):
        grid = candidates[start : start + chunk]
        pred = scores >= grid[:, None]
        fp = np.count_nonzero(pred & y_neg, axis=1)
        tp = np.count_nonzero(pred & y_pos, axis=1)
        fn = n_pos - tp
        costs[start : start + chunk] = cost_fp * fp + cost_fn * fn
    min_cost = float(costs.min())
    tied = np.flatnonzero(costs == min_cost)
    chosen = int(tied[tied.size // 2])
    return float(candidates[chosen]), min_cost


def psi_from_counts(reference_counts: np.ndarray, actual_counts: np.ndarray) -> float:
    """PSI = sum (actual - reference) * ln(actual / reference).

    Empty bins are clipped to 1e-6 and both distributions are renormalized.
    """
    ref = np.asarray(reference_counts, dtype=np.float64)
    act = np.asarray(actual_counts, dtype=np.float64)
    if ref.shape != act.shape:
        raise ValueError("count vectors must match")
    if ref.sum() <= 0 or act.sum() <= 0:
        raise ValueError("counts must be positive")
    ref = np.maximum(ref, 1e-6)
    act = np.maximum(act, 1e-6)
    ref = ref / ref.sum()
    act = act / act.sum()
    return float(np.sum((act - ref) * np.log(act / ref)))


def histogram_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(np.asarray(values, dtype=np.float64), bins=np.asarray(edges, dtype=np.float64))
    return counts.astype(np.int64)


def quantile_edges(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Monotone bin edges. Collapsed quantiles are dropped so empty bins are not invented."""
    reference = np.asarray(reference, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(reference, qs))
    if edges.size < 2:
        lo = float(reference.min()) if reference.size else 0.0
        edges = np.array([lo - 1.0, lo + 1.0], dtype=np.float64)
    edges = edges.astype(np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def psi_numeric(reference: np.ndarray, actual: np.ndarray, edges: np.ndarray | None = None) -> float:
    if edges is None:
        edges = quantile_edges(reference)
    return psi_from_counts(histogram_counts(reference, edges), histogram_counts(actual, edges))


def psi_discrete(reference: np.ndarray, actual: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.int64)
    act = np.asarray(actual, dtype=np.int64)
    levels = int(max(ref.max(initial=0), act.max(initial=0))) + 1
    if levels < 1:
        levels = 1
    return psi_from_counts(
        np.bincount(ref, minlength=levels),
        np.bincount(act, minlength=levels),
    )


def chi2_homogeneity(counts_a: np.ndarray, counts_b: np.ndarray) -> float:
    """Pearson chi-square for two rows of category counts. Drops all-zero categories."""
    a = np.asarray(counts_a, dtype=np.float64)
    b = np.asarray(counts_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("count rows must match")
    keep = (a + b) > 0
    a = a[keep]
    b = b[keep]
    total = a + b
    grand = float(total.sum())
    if grand <= 0 or a.sum() <= 0 or b.sum() <= 0:
        return 0.0
    exp_a = total * (a.sum() / grand)
    exp_b = total * (b.sum() / grand)
    return float((((a - exp_a) ** 2) / exp_a).sum() + (((b - exp_b) ** 2) / exp_b).sum())


def evaluate_op(value: float, op: str, threshold: float, threshold_high: float | None = None) -> bool:
    if op == ">=":
        return float(value) >= float(threshold)
    if op == ">":
        return float(value) > float(threshold)
    if op == "<=":
        return float(value) <= float(threshold)
    if op == "<":
        return float(value) < float(threshold)
    if op == "between":
        if threshold_high is None:
            raise ValueError("between needs threshold_high")
        return float(threshold) <= float(value) <= float(threshold_high)
    raise ValueError(f"unknown op {op}")


def criterion(
    name: str,
    value: float,
    op: str,
    threshold: float,
    threshold_high: float | None = None,
) -> dict:
    passed = evaluate_op(value, op, threshold, threshold_high)
    return {
        "name": name,
        "value": value,
        "op": op,
        "threshold": threshold,
        "threshold_high": threshold_high,
        "passed": bool(passed),
    }
