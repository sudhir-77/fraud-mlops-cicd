"""PNG artifacts. Matplotlib uses the non-interactive backend."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve


def mean_abs_shap(shap_values: np.ndarray | list) -> np.ndarray:
    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        values = shap_values
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3:
        values = values[1] if values.shape[0] > 1 else values[0]
    return np.mean(np.abs(values), axis=0)


def save_pr_curve(y_true: np.ndarray, scores: np.ndarray, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, color="#1f4e79", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Out-of-time precision-recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_shap_bars(importances: np.ndarray, names: list[str], path: Path) -> None:
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(names)[order], importances[order], color="#1f4e79")
    ax.set_xlabel("Mean absolute SHAP")
    ax.set_title("TreeSHAP on a validation sample")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_drift_bars(weeks: list[int], max_psi: list[float], threshold: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#c45911" if week == 11 else "#1f4e79" for week in weeks]
    ax.bar([str(week) for week in weeks], max_psi, color=colors)
    ax.axhline(threshold, color="#833c0c", linestyle="--", label=f"PSI {threshold}")
    ax.set_xlabel("Week")
    ax.set_ylabel("Max PSI")
    ax.set_title("Drift alarm uses the tallest bar")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_score_histograms(
    scores_by_week: dict[int, np.ndarray],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for week, scores in scores_by_week.items():
        ax.hist(scores, bins=40, density=True, histtype="step", linewidth=1.5, label=f"week {week}")
    ax.set_xlabel("Model score")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution by test week")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
