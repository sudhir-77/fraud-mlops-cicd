"""Kafka-shaped consumer API over an in-process log.

No socket is opened. `bootstrap_servers` is accepted so call sites match
kafka-python, and the broker object is what the consumer actually reads.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class ConsumerRecord:
    topic: str
    partition: int
    offset: int
    key: str | None
    value: dict


@dataclass
class ColumnBatch:
    topic: str
    start: int
    end: int
    event_ids: np.ndarray
    features: np.ndarray
    week: np.ndarray
    acq_region: np.ndarray

    def __len__(self) -> int:
        return int(self.end - self.start)


class _Sent:
    def __init__(self, topic: str, offset: int) -> None:
        self.topic = topic
        self.offset = offset

    def get(self, timeout: float | None = None) -> _Sent:
        return self


class InProcessBroker:
    """Append-only topics. Group commits are independent of each other."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: dict[str, list[dict]] = {}
        self._columns: dict[str, dict[str, np.ndarray]] = {}
        self._commits: dict[tuple[str, str], int] = {}

    def produce(self, topic: str, value: dict, key: str | None = None) -> int:
        if not isinstance(value, dict) or "event_id" not in value:
            raise ValueError("row messages need an event_id")
        with self._lock:
            if topic in self._columns:
                raise RuntimeError(f"topic {topic} is columnar")
            rows = self._rows.setdefault(topic, [])
            offset = len(rows)
            rows.append({"key": key, "value": value})
            return offset

    def publish_columns(
        self,
        topic: str,
        event_id: np.ndarray,
        features: np.ndarray,
        week: np.ndarray,
        acq_region: np.ndarray,
    ) -> int:
        event_id = np.asarray(event_id, dtype=np.int64)
        features = np.asarray(features, dtype=np.float32)
        week = np.asarray(week)
        acq_region = np.asarray(acq_region)
        if features.ndim != 2 or features.shape[0] != event_id.shape[0]:
            raise ValueError("features must be (n, p) aligned with event_id")
        if week.shape[0] != event_id.shape[0] or acq_region.shape[0] != event_id.shape[0]:
            raise ValueError("column lengths must match")
        with self._lock:
            if topic in self._rows:
                raise RuntimeError(f"topic {topic} is a row log")
            if topic in self._columns:
                raise RuntimeError(f"topic {topic} already has a columnar log")
            self._columns[topic] = {
                "event_id": event_id,
                "features": np.ascontiguousarray(features),
                "week": week,
                "acq_region": acq_region,
            }
            return int(event_id.shape[0])

    def size(self, topic: str) -> int:
        with self._lock:
            if topic in self._columns:
                return int(self._columns[topic]["event_id"].shape[0])
            return len(self._rows.get(topic, []))

    def committed(self, group_id: str, topic: str) -> int | None:
        with self._lock:
            return self._commits.get((group_id, topic))

    def commit_offset(self, group_id: str, topic: str, offset: int) -> None:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        with self._lock:
            current = self._commits.get((group_id, topic), 0)
            if offset < current:
                raise RuntimeError("commit would move the group offset backwards")
            self._commits[(group_id, topic)] = int(offset)

    def slice_rows(self, topic: str, start: int, max_records: int) -> list[ConsumerRecord]:
        with self._lock:
            rows = self._rows.get(topic, [])
            stop = min(len(rows), start + max_records)
            out = []
            for offset in range(start, stop):
                item = rows[offset]
                out.append(
                    ConsumerRecord(
                        topic=topic,
                        partition=0,
                        offset=offset,
                        key=item["key"],
                        value=item["value"],
                    )
                )
            return out

    def slice_columns(self, topic: str, start: int, max_records: int) -> ColumnBatch | None:
        with self._lock:
            columns = self._columns.get(topic)
            if columns is None:
                raise RuntimeError(f"topic {topic} is not columnar")
            n = int(columns["event_id"].shape[0])
            if start >= n:
                return None
            stop = min(n, start + max_records)
            return ColumnBatch(
                topic=topic,
                start=start,
                end=stop,
                event_ids=columns["event_id"][start:stop],
                features=columns["features"][start:stop],
                week=columns["week"][start:stop],
                acq_region=columns["acq_region"][start:stop],
            )


class KafkaProducer:
    def __init__(self, broker: InProcessBroker, bootstrap_servers: str | None = None) -> None:
        self._broker = broker
        self.bootstrap_servers = bootstrap_servers

    def send(self, topic: str, value: dict | None = None, key: str | None = None) -> _Sent:
        if value is None:
            raise ValueError("value is required")
        offset = self._broker.produce(topic, value, key=key)
        return _Sent(topic, offset)

    def flush(self) -> None:
        return None


class KafkaConsumer:
    """Subset of kafka-python's KafkaConsumer: poll, commit, and iteration.

    Offsets advance on poll. A new consumer in the same group resumes at the
    last committed offset, so an uncommitted batch is delivered again.
    """

    def __init__(
        self,
        topic: str,
        broker: InProcessBroker,
        group_id: str = "fraud-scorer",
        enable_auto_commit: bool = False,
        auto_offset_reset: str = "earliest",
        bootstrap_servers: str | None = None,
        value_deserializer=None,
    ) -> None:
        if auto_offset_reset not in {"earliest", "latest"}:
            raise ValueError("auto_offset_reset must be earliest or latest")
        self.topic = topic
        self.group_id = group_id
        self.enable_auto_commit = enable_auto_commit
        self.auto_offset_reset = auto_offset_reset
        self.bootstrap_servers = bootstrap_servers
        self.value_deserializer = value_deserializer
        self._broker = broker
        stored = broker.committed(group_id, topic)
        if stored is None:
            self._next = broker.size(topic) if auto_offset_reset == "latest" else 0
        else:
            self._next = stored

    def poll(self, timeout_ms: int = 0, max_records: int = 500) -> list[ConsumerRecord]:
        records = self._broker.slice_rows(self.topic, self._next, max_records)
        if self.value_deserializer is not None:
            decoded = []
            for record in records:
                value = record.value
                if isinstance(value, (bytes, bytearray)):
                    value = self.value_deserializer(value)
                decoded.append(
                    ConsumerRecord(record.topic, record.partition, record.offset, record.key, value)
                )
            records = decoded
        if records:
            self._next = records[-1].offset + 1
        if self.enable_auto_commit:
            self.commit()
        return records

    def poll_batch(self, max_records: int = 8192) -> ColumnBatch | None:
        batch = self._broker.slice_columns(self.topic, self._next, max_records)
        if batch is None:
            return None
        self._next = batch.end
        if self.enable_auto_commit:
            self.commit()
        return batch

    def commit(self) -> None:
        self._broker.commit_offset(self.group_id, self.topic, self._next)

    def __iter__(self):
        while True:
            records = self.poll(max_records=500)
            if not records:
                return
            yield from records


def remember_first(ledger: dict[int, float], event_id: int, score: float) -> bool:
    """Keep the first score for an event id. Later deliveries do not replace it."""
    if event_id in ledger:
        return False
    ledger[event_id] = float(score)
    return True
