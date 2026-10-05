"""Point-in-time feature values on a 5-row example, plus a future-row diff."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import MODEL_FEATURES
from src.features import FEATURE_SQL
from src.pit import count_future_leaks, feature_frame, probe_transactions


def test_feature_sql_does_not_use_mule_flag() -> None:
    assert "is_mule" not in FEATURE_SQL
    assert "mule_merchants" not in FEATURE_SQL
    forbidden = {"is_fraud", "fraud_type", "is_mule", "acq_region"}
    assert forbidden.isdisjoint(MODEL_FEATURES)


def test_hand_computed_features() -> None:
    frame = feature_frame(probe_transactions()).set_index("txn_id")
    expected = {
        1: dict(
            cust_txn_count_1h=0,
            cust_txn_count_24h=0,
            cust_amount_24h=0,
            cust_prior_mean_amount=0,
            amount_over_prior_mean=1,
            cust_txn_count_7d=0,
            card_txn_count_1h=0,
            device_txn_count_24h=0,
            device_prior_customers=0,
            merchant_txn_count_7d=0,
            merchant_fraud_rate_7d=0,
            merchant_mean_amount_7d=0,
            is_new_device=0,
            amount=10,
        ),
        2: dict(
            cust_txn_count_1h=1,
            cust_txn_count_24h=1,
            cust_amount_24h=10,
            cust_prior_mean_amount=10,
            amount_over_prior_mean=1.2,
            cust_txn_count_7d=1,
            card_txn_count_1h=1,
            device_txn_count_24h=1,
            device_prior_customers=1,
            merchant_txn_count_7d=1,
            merchant_fraud_rate_7d=0,
            merchant_mean_amount_7d=10,
            is_new_device=0,
            amount=12,
        ),
        3: dict(
            cust_txn_count_1h=0,
            cust_txn_count_24h=2,
            cust_amount_24h=22,
            cust_prior_mean_amount=11,
            amount_over_prior_mean=100 / 11,
            cust_txn_count_7d=2,
            card_txn_count_1h=0,
            device_txn_count_24h=0,
            device_prior_customers=0,
            merchant_txn_count_7d=0,
            merchant_fraud_rate_7d=0,
            merchant_mean_amount_7d=0,
            is_new_device=1,
            amount=100,
        ),
        4: dict(
            cust_txn_count_1h=0,
            cust_txn_count_24h=0,
            cust_amount_24h=0,
            cust_prior_mean_amount=0,
            amount_over_prior_mean=1,
            cust_txn_count_7d=0,
            card_txn_count_1h=0,
            device_txn_count_24h=2,
            device_prior_customers=1,
            merchant_txn_count_7d=2,
            merchant_fraud_rate_7d=0.5,
            merchant_mean_amount_7d=11,
            is_new_device=0,
            amount=50,
        ),
        5: dict(
            cust_txn_count_1h=1,
            cust_txn_count_24h=1,
            cust_amount_24h=50,
            cust_prior_mean_amount=50,
            amount_over_prior_mean=0.4,
            cust_txn_count_7d=1,
            card_txn_count_1h=1,
            device_txn_count_24h=3,
            device_prior_customers=2,
            merchant_txn_count_7d=3,
            merchant_fraud_rate_7d=1 / 3,
            merchant_mean_amount_7d=24,
            is_new_device=0,
            amount=20,
        ),
    }
    for txn_id, values in expected.items():
        row = frame.loc[txn_id]
        for name, target in values.items():
            assert math.isclose(float(row[name]), float(target), rel_tol=1e-6, abs_tol=1e-6), (
                name,
                txn_id,
                row[name],
                target,
            )
        assert math.isclose(float(row["log_amount"]), math.log1p(float(row["amount"])), rel_tol=1e-6)


def test_future_event_does_not_change_past_features() -> None:
    assert count_future_leaks() == 0


def test_splits_are_disjoint() -> None:
    from src.config import TEST_WEEKS, TRAIN_WEEKS, VAL_WEEKS

    assert set(TRAIN_WEEKS).isdisjoint(TEST_WEEKS)
    assert set(VAL_WEEKS).isdisjoint(TEST_WEEKS)
    assert set(TRAIN_WEEKS).isdisjoint(VAL_WEEKS)
    assert TEST_WEEKS == [10, 11, 12]
    assert 11 in TEST_WEEKS


def test_warehouse_dims_do_not_feed_features() -> None:
    from src.pit import probe_transactions
    from src.warehouse import build_star_schema, connect

    con = connect()
    try:
        build_star_schema(con, probe_transactions(), mule_ids=np.array([100], dtype=np.int32))
        n_fact = con.execute("SELECT COUNT(*) FROM fact_transactions").fetchone()[0]
        flags = con.execute(
            "SELECT merchant_id, is_mule FROM dim_merchant ORDER BY merchant_id"
        ).fetchdf()
        assert n_fact == 5
        mule = flags.set_index("merchant_id")["is_mule"].to_dict()
        assert int(mule[100]) == 1
        assert int(mule[200]) == 0
        columns = [row[0] for row in con.execute("DESCRIBE fact_transactions").fetchall()]
        assert "is_mule" not in columns
    finally:
        con.close()
    assert isinstance(probe_transactions(), pd.DataFrame)
