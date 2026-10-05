"""DuckDB star schema: fact_transactions plus customer, card, device, and merchant dims.

`is_mule` lives only on dim_merchant. Feature SQL reads fact_transactions and does
not join that flag.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def connect(path: Path | None = None, temp_dir: Path | None = None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:" if path is None else str(path))
    con.execute("SET threads=4")
    con.execute("SET memory_limit='1200MB'")
    con.execute("SET preserve_insertion_order=false")
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
        safe = temp_dir.as_posix().replace("'", "''")
        con.execute(f"SET temp_directory='{safe}'")
    return con


def _load_mule_ids(con: duckdb.DuckDBPyConnection, mule_ids: np.ndarray) -> None:
    con.execute("CREATE TABLE mule_merchants (merchant_id INTEGER)")
    if len(mule_ids) == 0:
        return
    mule_df = pd.DataFrame({"merchant_id": np.asarray(mule_ids, dtype=np.int32)})
    con.register("mule_df", mule_df)
    con.execute("INSERT INTO mule_merchants SELECT merchant_id FROM mule_df")
    con.unregister("mule_df")


def _build_dims(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE TABLE dim_customer AS SELECT DISTINCT customer_id FROM fact_transactions")
    con.execute(
        """
        CREATE TABLE dim_card AS
        SELECT DISTINCT card_id, customer_id FROM fact_transactions
        """
    )
    con.execute("CREATE TABLE dim_device AS SELECT DISTINCT device_id FROM fact_transactions")
    con.execute(
        """
        CREATE TABLE dim_merchant AS
        SELECT
            m.merchant_id,
            CASE WHEN mu.merchant_id IS NOT NULL THEN 1 ELSE 0 END AS is_mule,
            CAST(m.merchant_id % 8 AS INTEGER) AS category
        FROM (SELECT DISTINCT merchant_id FROM fact_transactions) m
        LEFT JOIN mule_merchants mu USING (merchant_id)
        """
    )


def build_star_schema(
    con: duckdb.DuckDBPyConnection,
    frame: pd.DataFrame,
    mule_ids: np.ndarray,
) -> None:
    con.register("raw_df", frame)
    con.execute("CREATE TABLE fact_transactions AS SELECT * FROM raw_df")
    con.unregister("raw_df")
    _load_mule_ids(con, mule_ids)
    _build_dims(con)


def build_star_schema_from_parquet(
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    mule_ids: np.ndarray,
) -> None:
    con.execute(
        "CREATE TABLE fact_transactions AS SELECT * FROM read_parquet(?)",
        [str(parquet_path)],
    )
    _load_mule_ids(con, mule_ids)
    _build_dims(con)
