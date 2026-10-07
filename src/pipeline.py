"""End-to-end run: simulate, feature SQL, train, score, serve, monitor.

Test weeks are written to data/splits.json before training. Rule tuning uses the
training matrix only. The cost threshold uses validation scores only.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import (
    ALERT_RATE,
    COST_FN,
    COST_FP,
    MODEL_FEATURES,
    MODEL_NAME,
    MONITOR_FEATURES,
    PROJECT_SLUG,
    PSI_ALERT,
    SEED,
    TEST_WEEKS,
    TRAIN_WEEKS,
    VAL_WEEKS,
    SimConfig,
    TrainConfig,
    full_config,
    quick_config,
    train_config_for,
)
from src.drift import (
    alerts_from_psi,
    drift_report_markdown,
    feature_psi_table,
    fraud_mix_chi2,
    score_psi_rows,
)
from src.features import build_features
from src.loadtest import serve_and_measure, summarize_latencies
from src.metrics import criterion, pr_auc, recall_at_alert_rate, select_threshold
from src.pit import count_future_leaks
from src.plots import (
    mean_abs_shap,
    save_drift_bars,
    save_pr_curve,
    save_score_histograms,
    save_shap_bars,
)
from src.registry import promote_model
from src.rules import rule_scores, tune_rules
from src.simulate import simulate, verify_transactions, write_transactions
from src.train import train_model
from src.warehouse import build_star_schema_from_parquet, connect


def log(message: str) -> None:
    print(message, flush=True)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_matrix(
    con,
    weeks: list[int],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    frames_x: list[np.ndarray] = []
    frames_y: list[np.ndarray] = []
    metas: list[pd.DataFrame] = []
    columns = ", ".join(feature_names)
    for week in weeks:
        frame = con.execute(
            f"""
            SELECT txn_id, week, day, is_fraud, {columns}
            FROM features
            WHERE week = ?
            ORDER BY txn_id
            """,
            [int(week)],
        ).fetchdf()
        frame = frame.sort_values("txn_id", kind="mergesort")
        frames_x.append(frame[feature_names].to_numpy(dtype=np.float32))
        frames_y.append(frame["is_fraud"].to_numpy(dtype=np.int8))
        metas.append(frame[["txn_id", "week", "day"]].copy())
        del frame
    return np.concatenate(frames_x), np.concatenate(frames_y), pd.concat(metas, ignore_index=True)


def _prediction_frame(
    meta: pd.DataFrame,
    y_true: np.ndarray,
    model_score: np.ndarray,
    baseline_score: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "txn_id": meta["txn_id"].to_numpy(dtype=np.int64),
            "week": meta["week"].to_numpy(dtype=np.int16),
            "day": meta["day"].astype(str).to_numpy(),
            "y_true": np.asarray(y_true, dtype=np.int8),
            "model_score": np.asarray(model_score, dtype=np.float64),
            "baseline_score": np.asarray(baseline_score, dtype=np.float64),
        }
    ).sort_values("txn_id")


def _score_matrix(booster, params, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    amount = matrix[:, MODEL_FEATURES.index("amount")]
    velocity = matrix[:, MODEL_FEATURES.index("cust_txn_count_1h")]
    is_new = matrix[:, MODEL_FEATURES.index("is_new_device")]
    baseline = rule_scores(amount, velocity, is_new, params)
    model = booster.predict(matrix, num_threads=1)
    return np.asarray(model, dtype=np.float64), np.asarray(baseline, dtype=np.float64)


def _shap_importance(booster, sample: np.ndarray) -> np.ndarray:
    import shap

    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(sample)
    importance = mean_abs_shap(values)
    if importance.shape[0] != sample.shape[1]:
        contrib = booster.predict(sample, pred_contrib=True, num_threads=1)
        importance = np.mean(np.abs(contrib[:, :-1]), axis=0)
    return importance


def _metrics_from_csv(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    y_true = frame["y_true"].to_numpy()
    model = frame["model_score"].to_numpy()
    baseline = frame["baseline_score"].to_numpy()
    model_ap = pr_auc(y_true, model)
    baseline_ap = pr_auc(y_true, baseline)
    recall = recall_at_alert_rate(
        y_true,
        model,
        frame["day"].astype(str).to_numpy(),
        frame["txn_id"].to_numpy(),
        ALERT_RATE,
    )
    baseline_recall = recall_at_alert_rate(
        y_true,
        baseline,
        frame["day"].astype(str).to_numpy(),
        frame["txn_id"].to_numpy(),
        ALERT_RATE,
    )
    return {
        "test_pr_auc": model_ap,
        "baseline_pr_auc": baseline_ap,
        "pr_auc_lift": float(model_ap / baseline_ap),
        "recall_at_1pct": recall,
        "baseline_recall_at_1pct": baseline_recall,
        "n_rows": float(len(frame)),
        "fraud_rate": float(np.mean(y_true)),
    }


def run_pipeline(
    root: Path,
    quick: bool = False,
    device: str = "cpu",
    sim_config: SimConfig | None = None,
    train_cfg: TrainConfig | None = None,
) -> dict:
    root = Path(root)
    data_dir = root / "data"
    results = root / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    config = sim_config if sim_config is not None else (quick_config() if quick else full_config())
    trees = train_cfg if train_cfg is not None else train_config_for(quick)

    log(f"simulating {config.n_transactions} transactions (quick={quick})")
    frame, mule_ids = simulate(config)
    parquet_path = data_dir / "transactions.parquet"
    write_transactions(frame, parquet_path)
    expected_rows = len(frame)
    del frame
    gc.collect()
    verify_transactions(parquet_path, expected_rows)
    np.save(data_dir / "mule_merchants.npy", mule_ids)

    manifest = {
        "train_weeks": list(TRAIN_WEEKS),
        "val_weeks": list(VAL_WEEKS),
        "test_weeks": list(TEST_WEEKS),
        "seed": SEED,
        "n_transactions": config.n_transactions,
    }
    if set(manifest["train_weeks"]) & set(manifest["test_weeks"]):
        raise RuntimeError("train and test weeks overlap")
    if set(manifest["val_weeks"]) & set(manifest["test_weeks"]):
        raise RuntimeError("validation and test weeks overlap")
    _write_json(data_dir / "splits.json", manifest)
    saved = json.loads((data_dir / "splits.json").read_text(encoding="utf-8"))
    train_weeks = [int(w) for w in saved["train_weeks"]]
    val_weeks = [int(w) for w in saved["val_weeks"]]
    test_weeks = [int(w) for w in saved["test_weeks"]]

    platform: dict = {}
    db_path = data_dir / "warehouse.duckdb"
    for stale in (db_path, Path(str(db_path) + ".wal")):
        if stale.exists():
            stale.unlink()
    con = connect(db_path, temp_dir=data_dir / "duckdb_tmp")
    try:
        log("building star schema and point-in-time features")
        build_star_schema_from_parquet(con, parquet_path, mule_ids)
        build_features(con)
        pit_violations = count_future_leaks()
        split_counts = {}
        for name, weeks in ("train", train_weeks), ("val", val_weeks), ("test", test_weeks):
            listed = ", ".join(str(w) for w in weeks)
            found = con.execute(f"SELECT COUNT(*) FROM features WHERE week IN ({listed})").fetchone()
            split_counts[name] = int(found[0])

        log("loading train and validation matrices")
        x_train, y_train, _meta_train = _load_matrix(con, train_weeks, MODEL_FEATURES)
        x_val, y_val, meta_val = _load_matrix(con, val_weeks, MODEL_FEATURES)
        log("tuning rules on training weeks")
        rule_params, _train_rule_ap = tune_rules(
            x_train[:, MODEL_FEATURES.index("amount")],
            x_train[:, MODEL_FEATURES.index("cust_txn_count_1h")],
            x_train[:, MODEL_FEATURES.index("is_new_device")],
            y_train,
        )
        log("training LightGBM")
        trained = train_model(x_train, y_train, x_val, y_val, MODEL_FEATURES, trees)
        model_dir = root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / "fraud_lgbm.txt"
        trained.save_model(str(model_file))
        booster = lgb.Booster(model_file=str(model_file))
        probe = x_val[: min(8, len(x_val))]
        if np.max(np.abs(trained.predict(probe, num_threads=1) - booster.predict(probe, num_threads=1))) > 1e-5:
            raise RuntimeError("reloaded model scores do not match the trained booster")
        del trained
        val_model, val_base = _score_matrix(booster, rule_params, x_val)
        threshold, val_cost = select_threshold(y_val, val_model, COST_FP, COST_FN)
        val_frame = _prediction_frame(meta_val, y_val, val_model, val_base)
        val_path = results / "val_predictions.csv"
        val_frame.to_csv(val_path, index=False)
        val_ap = pr_auc(y_val, val_model)

        rng = np.random.default_rng(SEED)
        n_examples = min(32, len(x_train))
        example_idx = rng.choice(len(x_train), size=n_examples, replace=False)
        examples = np.asarray(x_train[example_idx], dtype=np.float64)
        shap_n = min(trees.shap_rows, len(x_val))
        shap_idx = rng.choice(len(x_val), size=shap_n, replace=False)
        shap_sample = np.asarray(x_val[shap_idx], dtype=np.float64)

        del x_train, y_train, _meta_train
        gc.collect()

        log("scoring test weeks")
        test_parts: list[pd.DataFrame] = []
        for week in test_weeks:
            x_week, y_week, meta_week = _load_matrix(con, [week], MODEL_FEATURES)
            model_score, base_score = _score_matrix(booster, rule_params, x_week)
            test_parts.append(_prediction_frame(meta_week, y_week, model_score, base_score))
            del x_week, y_week, meta_week
        test_frame = pd.concat(test_parts, ignore_index=True).sort_values("txn_id")
        test_path = results / "test_predictions.csv"
        test_frame.to_csv(test_path, index=False)
        scored = _metrics_from_csv(test_path)

        log("drift")
        listed = ", ".join(str(w) for w in [val_weeks[0], *test_weeks])
        monitor_cols = ", ".join(["week", *MONITOR_FEATURES])
        monitor = con.execute(
            f"SELECT {monitor_cols} FROM features WHERE week IN ({listed})"
        ).fetchdf()
        feature_psi = feature_psi_table(monitor, val_weeks[0], test_weeks, MONITOR_FEATURES)
        scores_by_week = {
            int(week): test_frame.loc[test_frame["week"] == week, "model_score"].to_numpy()
            for week in test_weeks
        }
        score_psi = score_psi_rows(val_frame["model_score"].to_numpy(), scores_by_week)
        psi_table = pd.concat([feature_psi, score_psi], ignore_index=True)
        psi_path = results / "drift_weekly.csv"
        psi_table.to_csv(psi_path, index=False)
        psi_table = pd.read_csv(psi_path)
        alerts = alerts_from_psi(psi_table)
        mix = con.execute(
            """
            SELECT week, fraud_type, COUNT(*) AS n
            FROM features
            WHERE is_fraud = 1
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        ).fetchdf()
        mix_path = results / "fraud_mix_by_week.csv"
        mix.to_csv(mix_path, index=False)
        chi2 = fraud_mix_chi2(mix)
        (results / "drift_weekly.md").write_text(
            drift_report_markdown(psi_table, alerts, chi2),
            encoding="utf-8",
        )
        fraud_rows = con.execute("SELECT COUNT(*), SUM(is_fraud) FROM features").fetchone()
        log("streaming replay of the test weeks")
        from streaming.gate import allow_promotion, evaluate_production_and_degraded
        from streaming.replay import run_stream_replay

        platform["stream"] = run_stream_replay(
            con,
            booster,
            test_frame,
            val_frame["model_score"].to_numpy(dtype=np.float64),
            threshold,
            results,
        )
        log("model-quality gate on the validation week")
        platform["gate"] = evaluate_production_and_degraded(
            y_val,
            val_model,
            meta_val["day"].astype(str).to_numpy(),
            meta_val["txn_id"].to_numpy(),
        )
        _write_json(results / "gate_results.json", platform["gate"])
        if not allow_promotion(platform["gate"]["production"]):
            raise RuntimeError("production model failed the quality gate")
        if allow_promotion(platform["gate"]["degraded"]):
            raise RuntimeError("degraded model was not blocked")
    finally:
        con.close()

    from streaming.dockerlint import lint_dockerfile
    from streaming.manifests import manifests_ok, validate_manifest_dir

    package_root = Path(__file__).resolve().parents[1]
    log("validating Kubernetes manifests and the Dockerfile")
    manifest_report = validate_manifest_dir(package_root / "k8s")
    _write_json(results / "manifest_validation.json", manifest_report)
    platform["manifests_ok"] = manifests_ok(manifest_report)
    docker_problems = lint_dockerfile(package_root / "Dockerfile")
    platform["dockerfile_problems"] = docker_problems
    if docker_problems:
        raise RuntimeError("Dockerfile lint failed: " + "; ".join(docker_problems))
    if not platform["manifests_ok"]:
        raise RuntimeError("Kubernetes manifests failed schema validation for v1.29")

    payloads = []
    for i, row in enumerate(examples):
        payloads.append(
            {
                "txn_id": int(i + 1),
                "features": {name: float(row[j]) for j, name in enumerate(MODEL_FEATURES)},
            }
        )
    _write_json(
        model_dir / "serving.json",
        {
            "model_path": "models/fraud_lgbm.txt",
            "threshold": threshold,
            "feature_names": MODEL_FEATURES,
            "model_name": MODEL_NAME,
        },
    )

    log("registering model")
    promotion = promote_model(
        booster,
        root / "mlruns",
        {
            "val_pr_auc": val_ap,
            "test_pr_auc": scored["test_pr_auc"],
            "threshold": threshold,
        },
        {
            "num_trees": int(booster.num_trees()),
            "threshold": threshold,
            "seed": SEED,
        },
    )
    if promotion["stage"] != "Production":
        raise RuntimeError(f"expected Production stage, found {promotion['stage']}")

    log("plots")
    save_pr_curve(
        test_frame["y_true"].to_numpy(),
        test_frame["model_score"].to_numpy(),
        results / "pr_curve.png",
    )
    importance = _shap_importance(booster, shap_sample)
    save_shap_bars(importance, MODEL_FEATURES, results / "shap_importance.png")
    week_order = list(test_weeks)
    max_psi = []
    for week in week_order:
        group = psi_table[psi_table["week"] == week]
        max_psi.append(float(group["psi"].max()))
    save_drift_bars(week_order, max_psi, PSI_ALERT, results / "drift_psi.png")
    save_score_histograms(scores_by_week, results / "score_by_week.png")

    del test_frame, val_frame, x_val, y_val, shap_sample, monitor
    gc.collect()

    log(f"load test at {trees.load_rps} RPS for {trees.load_seconds}s")
    latencies = serve_and_measure(root, payloads, rps=trees.load_rps, seconds=trees.load_seconds)
    latency_frame = pd.DataFrame({"request_id": np.arange(len(latencies)), "latency_ms": latencies})
    latency_path = results / "loadtest_latencies.csv"
    latency_frame.to_csv(latency_path, index=False)
    latency_stats = summarize_latencies(
        pd.read_csv(latency_path)["latency_ms"].to_numpy()
    )

    week11 = psi_table[psi_table["week"] == 11]
    week11_top = week11.sort_values("psi", ascending=False).iloc[0]
    drift_ok = bool(alerts.get(11, False) and not alerts.get(10, True) and not alerts.get(12, True))
    stream = platform["stream"]
    gate = platform["gate"]
    headline = {
        "n_transactions": int(fraud_rows[0]),
        "test_pr_auc": round(scored["test_pr_auc"], 4),
        "pr_auc_lift": round(scored["pr_auc_lift"], 2),
        "recall_at_1pct": round(scored["recall_at_1pct"], 4),
        "p95_ms": round(latency_stats["p95_ms"], 2),
        "week_11_max_psi": round(float(week11["psi"].max()), 4),
        "stream_events_per_s": round(float(stream["events_per_s"]), 2),
    }
    from streaming.parent_check import load_parent_reference, max_relative_error, relative_error

    reread = _metrics_from_csv(test_path)
    psi_reread = pd.read_csv(psi_path)
    lat_reread = summarize_latencies(pd.read_csv(latency_path)["latency_ms"].to_numpy())
    artifact_error = max(
        relative_error(reread["test_pr_auc"], scored["test_pr_auc"]),
        relative_error(reread["recall_at_1pct"], scored["recall_at_1pct"]),
        relative_error(
            float(psi_reread.loc[psi_reread["week"] == 11, "psi"].max()),
            float(week11["psi"].max()),
        ),
        relative_error(lat_reread["p95_ms"], latency_stats["p95_ms"]),
    )
    if quick:
        parent_error = artifact_error
    else:
        parent_error = max(artifact_error, max_relative_error(headline, load_parent_reference(root)))
    gate_ok = bool(gate["production"]["passed"]) and not bool(gate["degraded"]["passed"])
    criteria = [
        criterion("Test PR-AUC (weeks 10-12)", scored["test_pr_auc"], ">=", 0.60),
        criterion("Test PR-AUC lift vs rules baseline", scored["pr_auc_lift"], ">=", 1.5),
        criterion("Recall at 1% alert rate", scored["recall_at_1pct"], ">=", 0.70),
        criterion("API p95 latency at 50 RPS", latency_stats["p95_ms"], "<", 50.0),
        criterion("Drift detection", 1.0 if drift_ok else 0.0, ">=", 1.0),
        criterion("Point-in-time correctness", float(pit_violations), "<=", 0.0),
        criterion("Streaming vs. batch parity", 1.0 if stream["identical"] else 0.0, ">=", 1.0),
        criterion("Streaming throughput on CPU", float(stream["events_per_s"]), ">=", 2000.0),
        criterion("Model gate", 1.0 if gate_ok else 0.0, ">=", 1.0),
        criterion("Drift alert", 1.0 if stream["drift_only_on_shift"] else 0.0, ">=", 1.0),
        criterion("Manifests", 1.0 if platform["manifests_ok"] else 0.0, ">=", 1.0),
        criterion("Parent project", float(parent_error), "<=", 0.02),
    ]
    metrics = {
        "project": PROJECT_SLUG,
        "success_criteria": criteria,
        "metrics": {
            "model": {
                "test_pr_auc": scored["test_pr_auc"],
                "val_pr_auc": val_ap,
                "test_recall_at_1pct_alert": scored["recall_at_1pct"],
                "threshold": threshold,
                "val_expected_cost": val_cost,
                "cost_fp": COST_FP,
                "cost_fn": COST_FN,
                "num_trees": int(booster.num_trees()),
                "best_iteration": int(booster.best_iteration or booster.num_trees()),
            },
            "drift": {
                "psi_threshold": PSI_ALERT,
                "reference_week": int(val_weeks[0]),
                "week_10_alert": int(bool(alerts.get(10, False))),
                "week_11_alert": int(bool(alerts.get(11, False))),
                "week_12_alert": int(bool(alerts.get(12, False))),
                "week_11_max_psi": float(week11["psi"].max()),
                "week_11_max_feature": str(week11_top["feature"]),
                "fraud_mix_chi2_week10_vs_week11": chi2,
            },
            "serving": {
                "p50_ms": latency_stats["p50_ms"],
                "p95_ms": latency_stats["p95_ms"],
                "max_ms": latency_stats["max_ms"],
                "rps": trees.load_rps,
                "seconds": trees.load_seconds,
                "n_requests": int(latency_stats["n_requests"]),
                "mlflow_model": promotion["model_name"],
                "mlflow_version": promotion["version"],
                "mlflow_stage": promotion["stage"],
                "mlflow_run_id": promotion["run_id"],
                "promotion": ["Staging", "Production"],
            },
            "streaming": {
                "events": int(stream["events"]),
                "seconds": float(stream["seconds"]),
                "events_per_s": float(stream["events_per_s"]),
                "max_abs_score_diff": float(stream["max_abs_diff"]),
                "identical": int(bool(stream["identical"])),
                "drift_only_on_shift": int(bool(stream["drift_only_on_shift"])),
                "alert_rate": float(stream["alert_rate"]),
                "latency_p95_ms": float(stream["latency_p95_ms"]),
                "latency_alert": int(bool(stream["latency_alert"])),
                "alert_rate_alert": int(bool(stream["alert_rate_alert"])),
                "prometheus_scored_series": int("fraud_scored_events_total" in stream["prometheus"]),
            },
            "gate": {
                "production_passed": int(bool(gate["production"]["passed"])),
                "degraded_passed": int(bool(gate["degraded"]["passed"])),
                "production_pr_auc": float(gate["production"]["pr_auc"]),
                "production_recall_at_1pct": float(gate["production"]["recall_at_1pct"]),
                "degraded_pr_auc": float(gate["degraded"]["pr_auc"]),
                "degraded_recall_at_1pct": float(gate["degraded"]["recall_at_1pct"]),
            },
            "deploy": {
                "manifests_valid": int(bool(platform["manifests_ok"])),
                "dockerfile_lint_ok": 1,
                "k8s_version": "1.29",
            },
            "parent_recompute_max_rel_error": float(parent_error),
            "headline": headline,
            "data": {
                "n_transactions": int(fraud_rows[0]),
                "n_fraud": int(fraud_rows[1]),
                "fraud_rate": float(fraud_rows[1]) / float(fraud_rows[0]),
                "test_fraud_rate": scored["fraud_rate"],
                "pit_future_feature_violations": int(pit_violations),
                "device_requested": device,
                "quick": int(quick),
            },
        },
        "baseline": {
            "test_pr_auc": scored["baseline_pr_auc"],
            "test_recall_at_1pct_alert": scored["baseline_recall_at_1pct"],
            "rules": rule_params.as_dict(),
        },
        "splits": {"transactions": split_counts},
        "seeds": [SEED],
        "hardware": {
            "cpu": "",
            "ram_gb": 0,
            "gpu": None,
            "python": "",
            "os": "",
            "peak_rss_gb": 0,
        },
        "runtime_seconds": 0.0,
    }
    _write_json(results / "metrics.json", metrics)
    log(
        "test PR-AUC "
        f"{scored['test_pr_auc']:.4f} lift {scored['pr_auc_lift']:.3f} "
        f"recall {scored['recall_at_1pct']:.4f} p95 {latency_stats['p95_ms']:.2f} ms "
        f"drift_ok {drift_ok} stream {stream['events_per_s']:.1f} events/s "
        f"parity {stream['identical']} gate {gate_ok}"
    )
    return metrics
