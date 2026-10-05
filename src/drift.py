"""Weekly PSI against the validation week, plus a fraud-mix chi-square.

The alert rule is: any monitored feature, or the model score, has PSI > 0.2.
Validation week is the reference so a slow reputation ramp inside training does
not look like drift once the monitor is on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FRAUD_TYPE_NAMES, MONITOR_FEATURES, PSI_ALERT, SHIFT_WEEK
from src.metrics import chi2_homogeneity, psi_discrete, psi_numeric, quantile_edges


def feature_psi_table(
    frame: pd.DataFrame,
    reference_week: int,
    compare_weeks: list[int],
    features: list[str] | None = None,
) -> pd.DataFrame:
    features = list(features or MONITOR_FEATURES)
    rows: list[dict] = []
    reference = frame[frame["week"] == reference_week]
    for feature in features:
        ref_values = reference[feature].to_numpy()
        discrete = feature == "acq_region"
        edges = None if discrete else quantile_edges(ref_values)
        for week in compare_weeks:
            current = frame.loc[frame["week"] == week, feature].to_numpy()
            if discrete:
                psi = psi_discrete(ref_values, current)
            else:
                psi = psi_numeric(ref_values, current, edges)
            rows.append(
                {
                    "week": int(week),
                    "feature": feature,
                    "psi": float(psi),
                    "alert": int(psi > PSI_ALERT),
                }
            )
    return pd.DataFrame(rows)


def score_psi_rows(
    reference_scores: np.ndarray,
    scores_by_week: dict[int, np.ndarray],
) -> pd.DataFrame:
    edges = quantile_edges(reference_scores)
    rows = []
    for week, scores in scores_by_week.items():
        psi = psi_numeric(reference_scores, scores, edges)
        rows.append(
            {
                "week": int(week),
                "feature": "model_score",
                "psi": float(psi),
                "alert": int(psi > PSI_ALERT),
            }
        )
    return pd.DataFrame(rows)


def alerts_from_psi(psi_table: pd.DataFrame) -> dict[int, bool]:
    alerts: dict[int, bool] = {}
    for week, group in psi_table.groupby("week"):
        alerts[int(week)] = bool((group["psi"] > PSI_ALERT).any())
    return alerts


def fraud_mix_chi2(mix: pd.DataFrame, week_a: int = 10, week_b: int = SHIFT_WEEK) -> float:
    """Homogeneity of fraud-type counts. `mix` has week, fraud_type, n."""
    levels = sorted(int(v) for v in mix["fraud_type"].unique())
    def counts(week: int) -> np.ndarray:
        sub = mix[mix["week"] == week]
        lookup = {int(row.fraud_type): int(row.n) for row in sub.itertuples(index=False)}
        return np.array([lookup.get(level, 0) for level in levels], dtype=np.float64)

    return chi2_homogeneity(counts(week_a), counts(week_b))


def drift_report_markdown(
    psi_table: pd.DataFrame,
    alerts: dict[int, bool],
    chi2: float,
) -> str:
    lines = [
        "# Weekly drift report",
        "",
        f"Alert rule: PSI > {PSI_ALERT} on any monitored feature or on the model score.",
        "Reference window: validation week 9.",
        "",
        "| Week | Alert | Max PSI | Feature at max |",
        "| --- | --- | --- | --- |",
    ]
    for week in sorted(alerts):
        group = psi_table[psi_table["week"] == week]
        top = group.sort_values("psi", ascending=False).iloc[0]
        fired = "yes" if alerts[week] else "no"
        lines.append(
            f"| {week} | {fired} | {float(top['psi']):.4f} | {top['feature']} |"
        )
    lines.extend(
        [
            "",
            (
                "Week 11 changes the fraud-type mix (more card testing and device-farm "
                f"fraud) and routes every transaction to acquirer region 0. Chi-square "
                f"for the week 10 vs week 11 fraud-type counts is {chi2:.4f}."
            ),
            "",
            "Stable weeks 10 and 12 keep the training fraud mix and the balanced region mix.",
            "",
        ]
    )
    # Touch the names so a reader can map integer codes if they open the CSV.
    lines.append("Fraud type codes: " + ", ".join(
        f"{i}={name}" for i, name in enumerate(FRAUD_TYPE_NAMES)
    ) + ".")
    lines.append("")
    return "\n".join(lines)
