"""Prometheus series and the alert rules evaluated on the replay."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

DRIFT_PSI_LIMIT = 0.2
LATENCY_P95_MS_LIMIT = 50.0
ALERT_RATE_LOW = 0.001
ALERT_RATE_HIGH = 0.05


def drift_rule(psi: float, limit: float = DRIFT_PSI_LIMIT) -> bool:
    return float(psi) > float(limit)


def latency_rule(p95_ms: float, limit_ms: float = LATENCY_P95_MS_LIMIT) -> bool:
    return float(p95_ms) >= float(limit_ms)


def alert_rate_rule(
    rate: float,
    low: float = ALERT_RATE_LOW,
    high: float = ALERT_RATE_HIGH,
) -> bool:
    value = float(rate)
    return value < float(low) or value > float(high)


class ScoringMonitor:
    def __init__(self) -> None:
        registry = CollectorRegistry()
        self.registry = registry
        self.scored = Counter(
            "fraud_scored_events_total",
            "Events scored by the streaming loop",
            registry=registry,
        )
        self.duplicates = Counter(
            "fraud_duplicate_events_total",
            "Redelivered event ids skipped by the ledger",
            registry=registry,
        )
        self.latency = Histogram(
            "fraud_score_latency_seconds",
            "Per-event time inside a scoring batch",
            registry=registry,
            buckets=(0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5),
        )
        self.week_psi = Gauge(
            "fraud_week_psi",
            "Max PSI against the validation week",
            ["week"],
            registry=registry,
        )
        self.drift_alert = Gauge(
            "fraud_drift_alert",
            "1 when the PSI rule fires for that week",
            ["week"],
            registry=registry,
        )
        self.alert_rate = Gauge(
            "fraud_alert_rate",
            "Share of streamed scores at or above the decision threshold",
            registry=registry,
        )
        self.event_latency_ms: list[float] = []

    def observe_batch(self, n_events: int, seconds: float, n_new: int, n_dup: int) -> None:
        if n_new:
            self.scored.inc(n_new)
        if n_dup:
            self.duplicates.inc(n_dup)
        per_event = float(seconds) / float(max(n_events, 1))
        self.latency.observe(per_event)
        self.event_latency_ms.append(per_event * 1000.0)

    def set_week(self, week: int, psi: float, alert: int) -> None:
        label = str(int(week))
        self.week_psi.labels(week=label).set(float(psi))
        self.drift_alert.labels(week=label).set(int(alert))

    def set_alert_rate(self, rate: float) -> None:
        self.alert_rate.set(float(rate))

    def render(self) -> str:
        return generate_latest(self.registry).decode("utf-8")
