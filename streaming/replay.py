"""Replay test-week events in event-time order and score them with the production booster.

The feature vector is the warehouse point-in-time row for that event. Batch
scoring uses the same rows and the same `predict(..., num_threads=1)` call, so
the scores are comparable elementwise. The in-process log is the only broker.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODEL_FEATURES, PSI_ALERT, SHIFT_WEEK, TEST_WEEKS, VAL_WEEKS
from src.metrics import psi_discrete, psi_numeric, quantile_edges
from src.plots import save_drift_bars
from streaming.broker import InProcessBroker, KafkaConsumer, remember_first
from streaming.monitor import ScoringMonitor, alert_rate_rule, drift_rule, latency_rule
from streaming.pit import gather_feature_rows

TOPIC = "transactions"
SCORE_BATCH = 8192


def _fetch_columns(con, weeks: list[int]) -> dict[str, np.ndarray]:
    feature_sql = ", ".join(MODEL_FEATURES)
    listed = ", ".join(str(int(week)) for week in weeks)
    query = f"""
        SELECT txn_id, week, acq_region, {feature_sql}
        FROM features
        WHERE week IN ({listed})
        ORDER BY event_ts, txn_id
    """
    table = con.execute(query).fetchnumpy()
    return {key: np.asarray(value) for key, value in table.items()}


def _matrix(table: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([np.asarray(table[name], dtype=np.float64) for name in MODEL_FEATURES])


def _align_batch_scores(test_frame: pd.DataFrame, event_ids: np.ndarray) -> np.ndarray:
    ids = test_frame["txn_id"].to_numpy(dtype=np.int64)
    scores = test_frame["model_score"].to_numpy(dtype=np.float64)
    order = np.argsort(ids, kind="mergesort")
    ids = ids[order]
    scores = scores[order]
    if ids.shape[0] != event_ids.shape[0]:
        raise RuntimeError("batch score file and stream log have different lengths")
    pos = np.searchsorted(ids, event_ids)
    if np.any(pos >= ids.shape[0]) or not np.array_equal(ids[pos], event_ids):
        raise RuntimeError("stream event ids do not match the batch score ids")
    return scores[pos]


def _weekly_max_psi(
    ref_matrix: np.ndarray,
    ref_region: np.ndarray,
    ref_scores: np.ndarray,
    week: np.ndarray,
    matrix: np.ndarray,
    region: np.ndarray,
    scores: np.ndarray,
) -> list[dict]:
    feature_edges = [quantile_edges(ref_matrix[:, index]) for index in range(ref_matrix.shape[1])]
    score_edges = quantile_edges(ref_scores)
    rows = []
    for test_week in TEST_WEEKS:
        mask = week == test_week
        if not np.any(mask):
            raise RuntimeError(f"stream log has no events for week {test_week}")
        values = [
            psi_numeric(ref_matrix[:, index], matrix[mask, index], feature_edges[index])
            for index in range(ref_matrix.shape[1])
        ]
        values.append(psi_discrete(ref_region, region[mask]))
        values.append(psi_numeric(ref_scores, scores[mask], score_edges))
        psi = float(max(values))
        shifted = int(test_week == SHIFT_WEEK)
        alert = int(drift_rule(psi, PSI_ALERT))
        rows.append(
            {
                "week": int(test_week),
                "psi": psi,
                "alert": alert,
                "shifted": shifted,
            }
        )
    return rows


def drift_only_on_shifted_week(rows: list[dict]) -> bool:
    saw_shift = False
    for row in rows:
        if int(row["shifted"]) == 1:
            saw_shift = True
            if int(row["alert"]) != 1:
                return False
        elif int(row["alert"]) != 0:
            return False
    return saw_shift


def consume_and_score(broker: InProcessBroker, booster, monitor: ScoringMonitor, batch_size: int = SCORE_BATCH):
    """Poll the columnar log, score each event once, and commit after each batch."""
    n = broker.size(TOPIC)
    consumer = KafkaConsumer(
        TOPIC,
        broker=broker,
        group_id="fraud-scorer",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        bootstrap_servers="in-process",
    )
    scores = np.empty(n, dtype=np.float64)
    ledger: dict[int, float] = {}
    started = time.perf_counter()
    while True:
        batch_started = time.perf_counter()
        batch = consumer.poll_batch(max_records=batch_size)
        if batch is None:
            break
        attached = gather_feature_rows(batch.event_ids, batch.features, batch.event_ids)
        pred = np.asarray(
            booster.predict(np.ascontiguousarray(attached, dtype=np.float32), num_threads=1),
            dtype=np.float64,
        )
        n_new = 0
        n_dup = 0
        for index, event_id in enumerate(batch.event_ids.tolist()):
            eid = int(event_id)
            if remember_first(ledger, eid, float(pred[index])):
                scores[batch.start + index] = pred[index]
                n_new += 1
            else:
                n_dup += 1
        consumer.commit()
        monitor.observe_batch(len(batch), time.perf_counter() - batch_started, n_new, n_dup)
    seconds = time.perf_counter() - started
    if len(ledger) != n:
        raise RuntimeError(f"ledger has {len(ledger)} events, log has {n}")
    return scores, n, seconds


def run_stream_replay(
    con,
    booster,
    test_frame: pd.DataFrame,
    val_scores: np.ndarray,
    threshold: float,
    results: Path,
) -> dict:
    results = Path(results)
    print("loading test-week events in event-time order", flush=True)
    streamed = _fetch_columns(con, list(TEST_WEEKS))
    reference = _fetch_columns(con, list(VAL_WEEKS))
    event_ids = np.asarray(streamed["txn_id"], dtype=np.int64)
    week = np.asarray(streamed["week"], dtype=np.int16)
    region = np.asarray(streamed["acq_region"], dtype=np.int64)
    matrix = _matrix(streamed)
    ref_matrix = _matrix(reference)
    ref_region = np.asarray(reference["acq_region"], dtype=np.int64)
    del streamed, reference
    if np.unique(event_ids).shape[0] != event_ids.shape[0]:
        raise RuntimeError("duplicate event ids in the test-week feature store")
    features = gather_feature_rows(event_ids, np.ascontiguousarray(matrix, dtype=np.float32), event_ids)
    broker = InProcessBroker()
    broker.publish_columns(TOPIC, event_ids, features, week, region)
    monitor = ScoringMonitor()
    stream_scores, n_events, seconds = consume_and_score(broker, booster, monitor)
    batch_scores = _align_batch_scores(test_frame, event_ids)
    identical = bool(np.array_equal(batch_scores, stream_scores))
    max_abs = float(np.max(np.abs(batch_scores - stream_scores))) if n_events else 0.0
    frame = pd.DataFrame(
        {
            "event_id": event_ids,
            "batch_score": batch_scores,
            "stream_score": stream_scores,
        }
    )
    frame.to_csv(results / "stream_vs_batch.csv", index=False)
    (results / "stream_throughput.json").write_text(
        json.dumps({"events": int(n_events), "seconds": float(seconds)}),
        encoding="utf-8",
    )
    rows = _weekly_max_psi(
        ref_matrix,
        ref_region,
        np.asarray(val_scores, dtype=np.float64),
        week,
        matrix,
        region,
        stream_scores,
    )
    drift = pd.DataFrame(rows)
    drift.to_csv(results / "stream_drift_weekly.csv", index=False)
    save_drift_bars(
        [int(row["week"]) for row in rows],
        [float(row["psi"]) for row in rows],
        PSI_ALERT,
        results / "stream_drift_psi.png",
    )
    for row in rows:
        monitor.set_week(int(row["week"]), float(row["psi"]), int(row["alert"]))
    rate = float(np.mean(stream_scores >= float(threshold)))
    monitor.set_alert_rate(rate)
    latency_p95 = float(np.percentile(np.asarray(monitor.event_latency_ms, dtype=np.float64), 95))
    del matrix, ref_matrix, features, broker
    return {
        "identical": identical,
        "max_abs_diff": max_abs,
        "events": int(n_events),
        "seconds": float(seconds),
        "events_per_s": float(n_events / seconds) if seconds > 0 else 0.0,
        "drift_only_on_shift": drift_only_on_shifted_week(rows),
        "alert_rate": rate,
        "latency_p95_ms": latency_p95,
        "latency_alert": bool(latency_rule(latency_p95)),
        "alert_rate_alert": bool(alert_rate_rule(rate)),
        "prometheus": monitor.render(),
    }
