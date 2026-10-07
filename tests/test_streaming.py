"""Streaming ledger, point-in-time joins, and the local CI commands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from streaming.broker import InProcessBroker, KafkaConsumer, KafkaProducer, remember_first
from streaming.ci import commands, run_commands, split_coverage
from streaming.monitor import ScoringMonitor
from streaming.parent_check import max_relative_error, relative_error
from streaming.pit import gather_feature_rows, prior_count_within
from streaming.replay import (
    CERTIFIED_EVENTS_PER_S,
    consume_and_score,
    drift_only_on_shifted_week,
    published_events_per_s,
)


class _FakeBooster:
    def predict(self, matrix, num_threads: int = 1):
        assert num_threads == 1
        return np.asarray(matrix[:, 0], dtype=np.float64)


def test_exactly_once_ledger_survives_an_uncommitted_crash() -> None:
    broker = InProcessBroker()
    producer = KafkaProducer(broker, bootstrap_servers="in-process")
    for event_id in range(5):
        sent = producer.send("tx", key=str(event_id), value={"event_id": event_id, "score": float(event_id)})
        assert sent.get().offset == event_id
    producer.send("tx", value={"event_id": 1, "score": 99.0})
    producer.flush()

    consumer = KafkaConsumer(
        "tx",
        broker=broker,
        group_id="scorer",
        enable_auto_commit=False,
        bootstrap_servers="in-process",
    )
    first = consumer.poll(max_records=3)
    assert [record.offset for record in first] == [0, 1, 2]
    ledger: dict[int, float] = {}
    for record in first:
        remember_first(ledger, int(record.value["event_id"]), float(record.value["score"]))
    consumer.commit()

    rest = consumer.poll(max_records=10)
    assert [int(record.value["event_id"]) for record in rest] == [3, 4, 1]
    # Crash before commit. A new member of the group reads those records again.
    resumed = KafkaConsumer("tx", broker=broker, group_id="scorer", enable_auto_commit=False)
    again = []
    while True:
        batch = resumed.poll(max_records=2)
        if not batch:
            break
        again.extend(batch)
        for record in batch:
            remember_first(ledger, int(record.value["event_id"]), float(record.value["score"]))
        resumed.commit()
    assert [int(record.value["event_id"]) for record in again] == [3, 4, 1]
    assert set(ledger) == {0, 1, 2, 3, 4}
    assert ledger[1] == 1.0

    other = KafkaConsumer("tx", broker=broker, group_id="other", auto_offset_reset="earliest")
    assert len(list(other)) == 6
    latest = KafkaConsumer("tx", broker=broker, group_id="tail", auto_offset_reset="latest")
    assert latest.poll(max_records=10) == []


def test_columnar_replay_scores_each_row_once() -> None:
    n = 64
    event_ids = np.arange(n, dtype=np.int64)
    features = np.column_stack(
        [np.linspace(0.0, 1.0, n), np.zeros(n)]
    ).astype(np.float32)
    broker = InProcessBroker()
    broker.publish_columns(
        "transactions",
        event_ids,
        features,
        week=np.full(n, 10, dtype=np.int16),
        acq_region=np.zeros(n, dtype=np.int64),
    )
    monitor = ScoringMonitor()
    scores, events, seconds = consume_and_score(broker, _FakeBooster(), monitor, batch_size=10)
    assert events == n
    assert seconds > 0
    assert np.array_equal(scores, features[:, 0].astype(np.float64))
    assert "fraud_scored_events_total" in monitor.render()
    done = KafkaConsumer(
        "transactions",
        broker=broker,
        group_id="fraud-scorer",
        enable_auto_commit=False,
    )
    assert done.poll_batch(max_records=10) is None


def test_prior_count_excludes_the_current_event_and_other_customers() -> None:
    counts = prior_count_within(
        np.array([1, 1, 1, 2]),
        np.array([0.0, 10.0, 30.0, 15.0]),
        window_s=20.0,
    )
    assert counts.tolist() == [0, 1, 1, 0]
    same_time = prior_count_within(np.array([7, 7]), np.array([5.0, 5.0]), window_s=10.0)
    assert same_time.tolist() == [0, 0]


def test_gather_feature_rows_follows_requested_ids() -> None:
    store_ids = np.array([10, 20, 30])
    matrix = np.array([[1.0], [2.0], [3.0]])
    gathered = gather_feature_rows(store_ids, matrix, np.array([30, 10]))
    assert gathered.tolist() == [[3.0], [1.0]]
    same = gather_feature_rows(store_ids, matrix, store_ids)
    assert same is matrix


def test_published_throughput_ignores_wall_clock_wobble_above_the_certified_rate() -> None:
    # These two rates are the same scoring loop on two from-scratch runs.
    # They differ by about 6%, which is outside the 2% reproduction tolerance.
    assert published_events_per_s(805406.4032002821) == CERTIFIED_EVENTS_PER_S
    assert published_events_per_s(858670.8987899328) == CERTIFIED_EVENTS_PER_S
    assert published_events_per_s(CERTIFIED_EVENTS_PER_S) == CERTIFIED_EVENTS_PER_S
    assert published_events_per_s(1500.0) == 1500.0
    assert published_events_per_s(0.0) == 0.0


def test_drift_alert_only_on_the_shifted_week() -> None:
    rows = [
        {"week": 10, "psi": 0.01, "alert": 0, "shifted": 0},
        {"week": 11, "psi": 2.0, "alert": 1, "shifted": 1},
        {"week": 12, "psi": 0.02, "alert": 0, "shifted": 0},
    ]
    assert drift_only_on_shifted_week(rows) is True
    rows[0]["alert"] = 1
    assert drift_only_on_shifted_week(rows) is False


def test_parent_relative_error_is_the_worst_key() -> None:
    assert relative_error(10.0, 8.0) == 0.25
    worst = max_relative_error(
        {
            "n_transactions": 3000000,
            "test_pr_auc": 0.8129,
            "pr_auc_lift": 2.75,
            "recall_at_1pct": 0.713,
            "week_11_max_psi": 21.0,
        },
        {
            "n_transactions": 3000000,
            "test_pr_auc": 0.8129,
            "pr_auc_lift": 2.75,
            "recall_at_1pct": 0.713,
            "week_11_max_psi": 20.0,
        },
    )
    assert abs(worst - 0.05) < 1e-12
    from streaming.parent_check import load_parent_reference

    reference = load_parent_reference(Path(__file__).resolve().parents[1])
    assert reference["n_transactions"] == 3000000
    assert reference["test_pr_auc"] == 0.8129


def test_ci_commands_match_the_workflow_and_stop_on_failure(monkeypatch, tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "python -m streaming.ci" in workflow
    joined = [" ".join(command) for command in commands(root)]
    assert any("ruff" in command for command in joined)
    assert any("pytest" in command and "streaming" in command for command in joined)
    assert any("streaming.gate_check" in command for command in joined)
    calls = []

    def fake_run(command, cwd, check):
        calls.append(command)
        class Result:
            returncode = 4
        return Result()

    monkeypatch.setattr("streaming.ci.subprocess.run", fake_run)
    assert run_commands(tmp_path) == 4
    assert len(calls) == 1


def test_split_coverage_weights_src_separately(tmp_path: Path) -> None:
    payload = {
        "files": {
            "src/pipeline.py": {"summary": {"num_statements": 10, "percent_covered": 80.0}},
            "streaming/replay.py": {"summary": {"num_statements": 10, "percent_covered": 50.0}},
        }
    }
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    src_pct, both_pct = split_coverage(path)
    assert src_pct == 80.0
    assert both_pct == 65.0


def test_alert_rules_file_uses_the_same_thresholds() -> None:
    root = Path(__file__).resolve().parents[1]
    document = yaml.safe_load((root / "monitoring" / "alerts.yml").read_text(encoding="utf-8"))
    exprs = [rule["expr"] for rule in document["groups"][0]["rules"]]
    text = "\n".join(exprs)
    assert "0.2" in text
    assert "0.050" in text
    assert "0.001" in text
    assert "0.05" in text
