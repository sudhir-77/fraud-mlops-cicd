"""Simulator invariants: size, fraud rate, week-11 shift, reproducibility, checksum."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SHIFT_WEEK, quick_config
from src.simulate import (
    allocate_mix,
    simulate,
    verify_transactions,
    write_transactions,
)


def test_allocate_full_mix_counts() -> None:
    normal = allocate_mix(3000, shifted=False)
    shifted = allocate_mix(3000, shifted=True)
    assert sum(normal.values()) == 3000
    assert sum(shifted.values()) == 3000
    assert normal == {"card_test": 600, "ato": 600, "mule": 900, "device_farm": 900}
    assert shifted["card_test"] > normal["card_test"]
    assert shifted["mule"] < normal["mule"]


def test_quick_simulation_invariants() -> None:
    config = quick_config()
    frame, mule_ids = simulate(config)
    assert len(frame) == config.n_transactions
    assert frame["txn_id"].is_unique
    assert set(frame["week"].unique()) == set(range(1, 13))
    assert int(frame.groupby("week").size().nunique()) == 1
    rate = float(frame["is_fraud"].mean())
    assert 0.01 <= rate <= 0.014
    week11 = frame[frame["week"] == SHIFT_WEEK]
    week10 = frame[frame["week"] == 10]
    assert int((week11["acq_region"] == 0).sum()) == len(week11)
    assert int(week10["acq_region"].nunique()) == 5
    fraud11 = week11[week11["is_fraud"] == 1]["fraud_type"].value_counts()
    fraud10 = week10[week10["is_fraud"] == 1]["fraud_type"].value_counts()
    assert int(fraud11.get(1, 0)) > int(fraud10.get(1, 0))
    assert set(np.asarray(mule_ids)).isdisjoint(set(range(config.n_merchants - config.n_mule_merchants)))
    card = frame[frame["fraud_type"] == 1]
    assert len(card) > 0
    gaps = card.sort_values("event_ts").groupby("customer_id")["event_ts"].diff().dt.total_seconds()
    assert (gaps.dropna() <= 120).any()


def test_fraud_overlaps_legitimate_behavior() -> None:
    config = quick_config()
    frame, mule_ids = simulate(config)
    mule_set = set(int(v) for v in np.asarray(mule_ids))
    legit = frame["is_fraud"] == 0
    fraud = frame["is_fraud"] == 1
    legit_merchants = set(frame.loc[legit, "merchant_id"].astype(int))
    assert mule_set & legit_merchants
    fraud_merchants = set(frame.loc[fraud, "merchant_id"].astype(int))
    assert fraud_merchants - mule_set
    legit_devices = set(frame.loc[legit, "device_id"].astype(int))
    fraud_devices = set(frame.loc[fraud, "device_id"].astype(int))
    assert legit_devices & fraud_devices
    # Disputes: labeled fraud with no injected pattern. Delayed chargebacks keep a pattern.
    assert int(((frame["is_fraud"] == 1) & (frame["fraud_type"] == 0)).sum()) > 0
    assert int(((frame["is_fraud"] == 0) & (frame["fraud_type"] != 0)).sum()) > 0


def test_simulation_is_reproducible() -> None:
    config = quick_config()
    first, _ = simulate(config)
    second, _ = simulate(config)
    assert np.array_equal(first["amount"].to_numpy(), second["amount"].to_numpy())
    assert np.array_equal(first["is_fraud"].to_numpy(), second["is_fraud"].to_numpy())


def test_parquet_checksum_and_row_count(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "txn_id": np.array([1, 2], dtype=np.int64),
            "amount": np.array([1.5, 2.5]),
        }
    )
    path = tmp_path / "transactions.parquet"
    write_transactions(frame, path)
    assert verify_transactions(path, 2)
    checksum = tmp_path / "transactions.parquet.sha256"
    checksum.write_text("deadbeef\n", encoding="utf-8")
    try:
        verify_transactions(path, 2)
        raised = False
    except ValueError:
        raised = True
    assert raised
