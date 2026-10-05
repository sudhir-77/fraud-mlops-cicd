"""Point-in-time features.

Every window stops one microsecond before the scored row, or uses
ROWS ... 1 PRECEDING. A feature never reads the current label or a later event.
Merchant fraud rate uses only earlier labels inside a 7-day window, which is the
warehouse analogue of a chargeback that has already landed. A lifetime count
grows every week and would trip PSI on a stable book.
"""

from __future__ import annotations

import duckdb

from src.config import MODEL_FEATURES

# is_mule and mule_merchants are intentionally absent. acq_region is passed
# through for the drift job and is not part of MODEL_FEATURES.
FEATURE_SQL = """
CREATE TABLE features AS
WITH base AS (
    SELECT
        txn_id,
        event_ts,
        week,
        CAST(strftime(event_ts, '%Y-%m-%d') AS VARCHAR) AS day,
        customer_id,
        card_id,
        device_id,
        merchant_id,
        amount,
        is_fraud,
        fraud_type,
        acq_region,
        CAST(EXTRACT(HOUR FROM event_ts) AS INTEGER) AS hour,
        LN(1 + amount) AS log_amount
    FROM fact_transactions
),
cust AS (
    SELECT
        txn_id,
        COUNT(*) OVER (
            PARTITION BY customer_id
            ORDER BY event_ts
            RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND INTERVAL 1 MICROSECOND PRECEDING
        ) AS cust_txn_count_1h,
        COUNT(*) OVER (
            PARTITION BY customer_id
            ORDER BY event_ts
            RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND INTERVAL 1 MICROSECOND PRECEDING
        ) AS cust_txn_count_24h,
        COALESCE(SUM(amount) OVER (
            PARTITION BY customer_id
            ORDER BY event_ts
            RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND INTERVAL 1 MICROSECOND PRECEDING
        ), 0) AS cust_amount_24h,
        COALESCE(AVG(amount) OVER (
            PARTITION BY customer_id
            ORDER BY event_ts
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0) AS cust_prior_mean_amount,
        COUNT(*) OVER (
            PARTITION BY customer_id
            ORDER BY event_ts
            RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND INTERVAL 1 MICROSECOND PRECEDING
        ) AS cust_txn_count_7d,
        COUNT(*) OVER (
            PARTITION BY customer_id
            ORDER BY event_ts
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS cust_prior_txn_count,
        COUNT(*) OVER (
            PARTITION BY customer_id, device_id
            ORDER BY event_ts
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS device_cust_prior_txn_count
    FROM base
),
card_w AS (
    SELECT
        txn_id,
        COUNT(*) OVER (
            PARTITION BY card_id
            ORDER BY event_ts
            RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND INTERVAL 1 MICROSECOND PRECEDING
        ) AS card_txn_count_1h
    FROM base
),
device_w AS (
    SELECT
        txn_id,
        COUNT(*) OVER (
            PARTITION BY device_id
            ORDER BY event_ts
            RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND INTERVAL 1 MICROSECOND PRECEDING
        ) AS device_txn_count_24h
    FROM base
),
merchant_w AS (
    SELECT
        txn_id,
        COALESCE(SUM(is_fraud) OVER (
            PARTITION BY merchant_id
            ORDER BY event_ts
            RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND INTERVAL 1 MICROSECOND PRECEDING
        ), 0) AS merchant_fraud_sum_7d,
        COUNT(*) OVER (
            PARTITION BY merchant_id
            ORDER BY event_ts
            RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND INTERVAL 1 MICROSECOND PRECEDING
        ) AS merchant_txn_count_7d,
        COALESCE(AVG(amount) OVER (
            PARTITION BY merchant_id
            ORDER BY event_ts
            RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND INTERVAL 1 MICROSECOND PRECEDING
        ), 0) AS merchant_mean_amount_7d
    FROM base
),
device_first AS (
    SELECT device_id, customer_id, MIN(event_ts) AS first_ts
    FROM base
    GROUP BY 1, 2
),
device_cum AS (
    SELECT
        device_id,
        first_ts,
        ROW_NUMBER() OVER (PARTITION BY device_id ORDER BY first_ts) AS n_through
    FROM device_first
),
device_reuse AS (
    SELECT
        b.txn_id,
        COALESCE(d.n_through, 0) AS device_prior_customers
    FROM base b
    ASOF LEFT JOIN device_cum d
        ON b.device_id = d.device_id
       AND b.event_ts > d.first_ts
)
SELECT
    b.txn_id,
    b.event_ts,
    b.week,
    b.day,
    b.customer_id,
    b.card_id,
    b.device_id,
    b.merchant_id,
    b.amount,
    b.is_fraud,
    b.fraud_type,
    b.acq_region,
    b.hour,
    b.log_amount,
    c.cust_txn_count_1h,
    c.cust_txn_count_24h,
    c.cust_amount_24h,
    c.cust_prior_mean_amount,
    CASE
        WHEN c.cust_prior_mean_amount = 0 THEN 1.0
        ELSE b.amount / c.cust_prior_mean_amount
    END AS amount_over_prior_mean,
    c.cust_txn_count_7d,
    k.card_txn_count_1h,
    d.device_txn_count_24h,
    r.device_prior_customers,
    m.merchant_txn_count_7d,
    CASE
        WHEN m.merchant_txn_count_7d = 0 THEN 0.0
        ELSE m.merchant_fraud_sum_7d / m.merchant_txn_count_7d
    END AS merchant_fraud_rate_7d,
    m.merchant_mean_amount_7d,
    CASE
        WHEN c.cust_prior_txn_count > 0 AND c.device_cust_prior_txn_count = 0 THEN 1
        ELSE 0
    END AS is_new_device
FROM base b
JOIN cust c USING (txn_id)
JOIN card_w k USING (txn_id)
JOIN device_w d USING (txn_id)
JOIN merchant_w m USING (txn_id)
JOIN device_reuse r USING (txn_id)
"""


def build_features(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DROP TABLE IF EXISTS features")
    con.execute(FEATURE_SQL)
    columns = [row[0] for row in con.execute("DESCRIBE features").fetchall()]
    missing = [name for name in MODEL_FEATURES if name not in columns]
    if missing:
        raise RuntimeError(f"feature table missing columns: {missing}")
