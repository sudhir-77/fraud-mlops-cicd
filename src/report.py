"""README results table and resume notes, filled from metrics.json numbers."""

from __future__ import annotations

import json
from pathlib import Path

from src.metrics import criterion


def _token(metrics: dict, *path: str) -> str:
    value: object = metrics
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(path)
        value = value[key]
    return json.dumps(value)


def finalize_metrics(
    metrics: dict,
    host: dict,
    peak_rss_gb: float,
    runtime_seconds: float,
    coverage_percent: float | None,
) -> dict:
    metrics["hardware"] = {
        "cpu": host.get("cpu", ""),
        "ram_gb": host.get("ram_gb", 0),
        "gpu": None,
        "python": host.get("python", ""),
        "os": host.get("os", ""),
        "peak_rss_gb": peak_rss_gb,
    }
    metrics["runtime_seconds"] = runtime_seconds
    kept = [item for item in metrics["success_criteria"] if item["name"] != "Test coverage of src/"]
    if coverage_percent is not None:
        kept.append(criterion("Test coverage of src/", float(coverage_percent), ">=", 80.0))
        metrics["metrics"]["coverage_percent"] = float(coverage_percent)
    metrics["success_criteria"] = kept
    return metrics


def render_results_table(metrics: dict) -> str:
    lines = [
        "| Criterion | Value | Op | Threshold | Passed |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in metrics["success_criteria"]:
        high = item.get("threshold_high")
        threshold = item["threshold"] if high is None else f"{item['threshold']} to {high}"
        passed = "yes" if item["passed"] else "no"
        lines.append(
            f"| {item['name']} | {item['value']} | {item['op']} | {threshold} | {passed} |"
        )
    headline = metrics["metrics"].get("headline", {})
    if headline:
        lines.extend(
            [
                "",
                "Headline figures stored in `metrics.headline`:",
                "",
                "| Figure | Value |",
                "| --- | --- |",
            ]
        )
        for key, value in headline.items():
            lines.append(f"| {key} | {value} |")
    failed = [item["name"] for item in metrics["success_criteria"] if not item["passed"]]
    lines.append("")
    if failed:
        lines.append("Criteria that failed: " + ", ".join(failed) + ".")
    else:
        lines.append("All success criteria passed.")
    lines.append("")
    return "\n".join(lines)


def update_readme(readme_path: Path, metrics: dict) -> None:
    text = readme_path.read_text(encoding="utf-8")
    start = "<!-- RESULTS:START -->"
    end = "<!-- RESULTS:END -->"
    if start not in text or end not in text:
        raise RuntimeError("README is missing results markers")
    pre, rest = text.split(start, 1)
    _, post = rest.split(end, 1)
    table = render_results_table(metrics)
    readme_path.write_text(pre + start + "\n" + table + end + post, encoding="utf-8")


def write_resume_notes(path: Path, metrics: dict) -> None:
    """Bullets copy JSON number tokens from headline fields."""
    n = _token(metrics, "metrics", "headline", "n_transactions")
    prauc = _token(metrics, "metrics", "headline", "test_pr_auc")
    lift = _token(metrics, "metrics", "headline", "pr_auc_lift")
    p95 = _token(metrics, "metrics", "headline", "p95_ms")
    recall = _token(metrics, "metrics", "headline", "recall_at_1pct")
    psi = _token(metrics, "metrics", "headline", "week_11_max_psi")
    drift = metrics["metrics"]["drift"]
    shift_flagged = (
        int(drift["week_11_alert"]) == 1
        and int(drift["week_10_alert"]) == 0
        and int(drift["week_12_alert"]) == 0
    )
    if shift_flagged:
        drift_bullet = (
            f"Added PSI-based drift monitoring that flagged an injected fraud-pattern shift "
            f"in week 11 (peak feature PSI {psi}) and did not alert in weeks 10 or 12."
        )
    else:
        drift_bullet = (
            f"Added PSI-based drift monitoring on the simulated weeks "
            f"(week 11 peak feature PSI {psi})."
        )
    text = "\n".join(
        [
            (
                f"Built a point-in-time-correct SQL feature pipeline and LightGBM fraud model "
                f"on {n} simulated transactions, reaching {prauc} PR-AUC ({lift}x a rules baseline) "
                f"on out-of-time weeks."
            ),
            (
                f"Served the model through FastAPI with SHAP reason codes at {p95} ms p95 latency "
                f"under 50 RPS, with versions promoted through an MLflow model registry."
            ),
            drift_bullet,
            (
                f"Caught {recall} of out-of-time simulated fraud when reviewing the top 1% of "
                f"scores each day, with the review threshold chosen by expected cost on the validation week only."
            ),
            "",
            "Skills/tools: Python, SQL, DuckDB, LightGBM, scikit-learn, MLflow, FastAPI, SHAP, pytest.",
            "",
            (
                "This project simulates 12 weeks of card payments with card testing, account takeover, "
                "mule merchants, and shared-device fraud, plus legitimate lookalikes and label noise. "
                "It builds point-in-time SQL features, trains a LightGBM scorer against hand-written rules, "
                "serves scores with SHAP reason codes, and checks a later week for population drift."
            ),
            "",
            (
                "GitHub: Point-in-time fraud features, a cost-thresholded LightGBM scorer, "
                "FastAPI serving, and PSI drift monitoring on simulated payments."
            ),
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
