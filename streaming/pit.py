"""Point-in-time helpers for the stream.

The replay scores the warehouse feature row for each event. That row was built
with windows that stop before the event. `gather_feature_rows` only reads those
rows. `prior_count_within` is the small online counter the tests check by hand:
it counts earlier events for the same customer inside a time window and skips
the current event.
"""

from __future__ import annotations

import numpy as np


def gather_feature_rows(
    store_ids: np.ndarray,
    store_matrix: np.ndarray,
    event_ids: np.ndarray,
) -> np.ndarray:
    store_ids = np.asarray(store_ids, dtype=np.int64)
    event_ids = np.asarray(event_ids, dtype=np.int64)
    matrix = np.asarray(store_matrix)
    if matrix.shape[0] != store_ids.shape[0]:
        raise ValueError("feature matrix and store ids have different lengths")
    if store_ids.shape[0] == event_ids.shape[0] and np.array_equal(store_ids, event_ids):
        return matrix
    order = np.argsort(store_ids, kind="mergesort")
    sorted_ids = store_ids[order]
    pos = np.searchsorted(sorted_ids, event_ids)
    if np.any(pos >= sorted_ids.shape[0]) or not np.array_equal(sorted_ids[pos], event_ids):
        raise KeyError("event id missing from the point-in-time feature store")
    return matrix[order[pos]]


def prior_count_within(
    customer_id: np.ndarray,
    event_ts: np.ndarray,
    window_s: float,
) -> np.ndarray:
    """Earlier same-customer events with 0 < age <= window_s.

    Equal timestamps do not count. The result follows the input order.
    """
    customer_id = np.asarray(customer_id)
    event_ts = np.asarray(event_ts, dtype=np.float64)
    if customer_id.shape[0] != event_ts.shape[0]:
        raise ValueError("customer_id and event_ts must have the same length")
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    out = np.zeros(customer_id.shape[0], dtype=np.int64)
    history: dict[int, list[float]] = {}
    order = np.argsort(event_ts, kind="mergesort")
    for idx in order:
        cid = int(customer_id[idx])
        ts = float(event_ts[idx])
        prior = history.get(cid, [])
        kept = [stamp for stamp in prior if 0 < ts - stamp <= window_s]
        out[idx] = len(kept)
        prior.append(ts)
        history[cid] = [stamp for stamp in prior if ts - stamp <= window_s]
    return out
