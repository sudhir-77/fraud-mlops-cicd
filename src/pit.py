"""Small point-in-time probe used by tests and by the pipeline metric.

`count_future_leaks` rebuilds features after appending a later event. Any model
feature that changes on the original rows is a future leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MODEL_FEATURES
from src.features import build_features
from src.warehouse import build_star_schema, connect


def probe_transactions() -> pd.DataFrame:
    """Five hand-checked rows. Times are minutes from 2024-01-01 00:00."""
    t0 = np.datetime64("2024-01-01T00:00:00", "ns")
    minutes = np.array([0, 10, 120, 180, 210], dtype=np.int64)
    event_ts = t0 + minutes.astype("timedelta64[m]")
    return pd.DataFrame(
        {
            "txn_id": np.array([1, 2, 3, 4, 5], dtype=np.int64),
            "event_ts": event_ts,
            "week": np.ones(5, dtype=np.int8),
            "customer_id": np.array([1, 1, 1, 2, 2], dtype=np.int32),
            "card_id": np.array([1, 1, 1, 2, 2], dtype=np.int32),
            "device_id": np.array([10, 10, 20, 10, 10], dtype=np.int32),
            "merchant_id": np.array([100, 100, 200, 100, 100], dtype=np.int32),
            "amount": np.array([10.0, 12.0, 100.0, 50.0, 20.0], dtype=np.float64),
            "is_fraud": np.array([0, 1, 0, 0, 1], dtype=np.int8),
            "fraud_type": np.array([0, 2, 0, 0, 3], dtype=np.int8),
            "acq_region": np.zeros(5, dtype=np.int8),
        }
    )


def future_event() -> pd.DataFrame:
    """Ten minutes after the last probe row, same customer, device, and merchant."""
    event_ts = np.datetime64("2024-01-01T00:00:00", "ns") + np.timedelta64(220, "m")
    return pd.DataFrame(
        {
            "txn_id": np.array([6], dtype=np.int64),
            "event_ts": np.array([event_ts], dtype="datetime64[ns]"),
            "week": np.array([1], dtype=np.int8),
            "customer_id": np.array([2], dtype=np.int32),
            "card_id": np.array([2], dtype=np.int32),
            "device_id": np.array([10], dtype=np.int32),
            "merchant_id": np.array([100], dtype=np.int32),
            "amount": np.array([999.0], dtype=np.float64),
            "is_fraud": np.array([1], dtype=np.int8),
            "fraud_type": np.array([3], dtype=np.int8),
            "acq_region": np.array([0], dtype=np.int8),
        }
    )


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    con = connect()
    try:
        build_star_schema(con, frame, mule_ids=np.array([100], dtype=np.int32))
        build_features(con)
        return con.execute("SELECT * FROM features ORDER BY txn_id").fetchdf()
    finally:
        con.close()


def count_future_leaks() -> int:
    base = probe_transactions()
    before = feature_frame(base).set_index("txn_id")
    after = feature_frame(pd.concat([base, future_event()], ignore_index=True)).set_index("txn_id")
    changed = 0
    for name in MODEL_FEATURES:
        left = before[name].to_numpy(dtype=np.float64)
        right = after.loc[before.index, name].to_numpy(dtype=np.float64)
        if not np.allclose(left, right, rtol=0.0, atol=1e-9, equal_nan=True):
            changed += 1
    return int(changed)
