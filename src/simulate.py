"""Simulated payment transactions.

Patterns, all labeled in the generator and not copied from a real feed:

- card_test: several charges on one card a few seconds apart. Some are tiny and
  dense. Others use ordinary amounts so they look like a legitimate burst.
- ato: a later transaction, sometimes on a new device for a large amount and
  sometimes on the customer's own device for a normal amount.
- mule: a normal-looking purchase. Some land on mule merchants that also take
  a lot of legitimate volume. Some land on popular shared merchants.
- device_farm: some fraud shares a small device pool that legitimate customers
  also use. The rest uses the customer's own device.

A fraction of fraud labels are withheld (delayed chargebacks) and the same
number of ordinary purchases are marked fraud (disputes). Week 11 keeps the
same fraud count but shifts the mix toward card testing and device farms, and
routes every transaction to acquirer region 0. Other weeks spread region evenly
across 0..4 so a PSI monitor has a stable baseline. Amounts and entity links
follow the spirit of PaySim/BankSim and are generated locally.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.config import (
    FRAUD_ATO,
    FRAUD_CARD,
    FRAUD_FARM,
    FRAUD_LEGIT,
    FRAUD_MULE,
    SHIFT_WEEK,
    SimConfig,
)

_BASE = np.datetime64("2024-01-01T00:00:00", "ns")
_WEEK_SECONDS = 7 * 24 * 3600
_FARM_DEVICE_BASE = 10_000_000
_ATO_DEVICE_BASE = 20_000_000
_SHARED_DEVICE_BASE = 30_000_000
_LEGIT_NEW_DEVICE_BASE = 40_000_000
_CARD_GROUP = 6
_LABEL_SWAP_FRACTION = 0.04
_CARD_QUIET_FRACTION = 0.05
_ATO_QUIET_FRACTION = 0.05
_MULE_ON_POPULAR_FRACTION = 0.05
_FARM_ON_OWN_DEVICE_FRACTION = 0.05


def allocate_mix(total: int, shifted: bool) -> dict[str, int]:
    if shifted:
        ratios = (
            ("card_test", 0.45),
            ("ato", 0.10),
            ("mule", 0.10),
            ("device_farm", 0.35),
        )
    else:
        ratios = (
            ("card_test", 0.20),
            ("ato", 0.20),
            ("mule", 0.30),
            ("device_farm", 0.30),
        )
    raw = [(name, total * weight) for name, weight in ratios]
    counts = {name: int(math.floor(value)) for name, value in raw}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda item: item[1] - math.floor(item[1]), reverse=True)
    for i in range(remainder):
        counts[order[i][0]] += 1
    if sum(counts.values()) != total:
        raise RuntimeError("fraud mix does not sum to the weekly fraud count")
    return counts


def validate_config(config: SimConfig) -> None:
    if config.n_weeks != 12:
        raise ValueError("the simulator is defined for 12 weeks")
    if config.n_per_week % 5 != 0:
        raise ValueError("n_per_week must be divisible by 5 so regions are balanced")
    if config.fraud_per_week >= config.n_per_week:
        raise ValueError("fraud_per_week must be smaller than n_per_week")
    if not 0 < config.n_mule_merchants < config.n_merchants:
        raise ValueError("mule merchants must be a non-empty subset of merchants")
    if config.n_farm_devices < 1:
        raise ValueError("need at least one farm device")
    if config.fraud_per_week > config.n_customers:
        raise ValueError("not enough customers for without-replacement fraud draws")


def mule_merchant_ids(config: SimConfig) -> np.ndarray:
    start = config.n_merchants - config.n_mule_merchants
    return np.arange(start, config.n_merchants, dtype=np.int32)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": np.array([], dtype=np.int32),
            "card_id": np.array([], dtype=np.int32),
            "device_id": np.array([], dtype=np.int32),
            "merchant_id": np.array([], dtype=np.int32),
            "amount": np.array([], dtype=np.float64),
            "offset_s": np.array([], dtype=np.int64),
            "is_fraud": np.array([], dtype=np.int8),
            "fraud_type": np.array([], dtype=np.int8),
        }
    )


def _frame_from_arrays(
    customer: np.ndarray,
    device: np.ndarray,
    merchant: np.ndarray,
    amount: np.ndarray,
    offset_s: np.ndarray,
    is_fraud: int,
    fraud_type: int,
) -> pd.DataFrame:
    customer = np.asarray(customer, dtype=np.int32)
    n = customer.shape[0]
    return pd.DataFrame(
        {
            "customer_id": customer,
            "card_id": customer.copy(),
            "device_id": np.asarray(device, dtype=np.int32),
            "merchant_id": np.asarray(merchant, dtype=np.int32),
            "amount": np.asarray(amount, dtype=np.float64),
            "offset_s": np.asarray(offset_s, dtype=np.int64),
            "is_fraud": np.full(n, is_fraud, dtype=np.int8),
            "fraud_type": np.full(n, fraud_type, dtype=np.int8),
        }
    )


def _n_popular(n_clean_merchants: int) -> int:
    return max(1, n_clean_merchants // 8)


def _n_shared_devices(config: SimConfig) -> int:
    return max(8, config.n_farm_devices * 4)


def _grouped_charges(
    rng: np.random.Generator,
    config: SimConfig,
    n_rows: int,
    max_offset: int,
    group_size: int,
    spacing_s: int,
    amount_low: float,
    amount_high: float,
    n_popular: int,
    n_clean_merchants: int,
    popular_fraction: float,
    shared_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Repeated charges for one customer. Spacing stays under two minutes."""
    if n_rows <= 0:
        empty_i = np.array([], dtype=np.int32)
        empty_a = np.array([], dtype=np.float64)
        empty_o = np.array([], dtype=np.int64)
        return empty_i, empty_i, empty_i, empty_a, empty_o
    n_groups = (n_rows + group_size - 1) // group_size
    sizes = np.full(n_groups, group_size, dtype=np.int32)
    sizes[-1] = n_rows - group_size * (n_groups - 1)
    starts_idx = np.cumsum(sizes) - sizes
    idx_in = np.arange(int(sizes.sum()), dtype=np.int64) - np.repeat(starts_idx, sizes)
    group_customer = rng.integers(0, config.n_customers, size=n_groups, dtype=np.int32)
    customer = np.repeat(group_customer, sizes)
    room = group_size * spacing_s + 1
    start = rng.integers(0, max(1, max_offset - room), size=n_groups, dtype=np.int64)
    offset_s = np.repeat(start, sizes) + idx_in * int(spacing_s)
    merchant_id = rng.integers(0, n_popular, size=n_groups, dtype=np.int32)
    n_other = int(round((1.0 - popular_fraction) * n_groups))
    if n_other > 0 and n_popular < n_clean_merchants:
        other = rng.choice(n_groups, size=n_other, replace=False)
        merchant_id[other] = rng.integers(
            n_popular, n_clean_merchants, size=n_other, dtype=np.int32
        )
    merchant = np.repeat(merchant_id, sizes)
    device = customer.copy()
    n_shared_groups = int(round(shared_fraction * n_groups))
    if n_shared_groups > 0:
        shared_groups = np.sort(rng.choice(n_groups, size=n_shared_groups, replace=False))
        shared_ids = (
            _SHARED_DEVICE_BASE + rng.integers(0, _n_shared_devices(config), size=n_shared_groups)
        ).astype(np.int32)
        group_of_row = np.repeat(np.arange(n_groups), sizes)
        device[np.isin(group_of_row, shared_groups)] = np.repeat(shared_ids, sizes[shared_groups])
    amount = rng.uniform(amount_low, amount_high, size=n_rows)
    return customer, device, merchant, amount, offset_s


def _card_test_frame(
    rng: np.random.Generator,
    config: SimConfig,
    n_card: int,
    max_offset: int,
    n_popular: int,
    n_clean_merchants: int,
) -> pd.DataFrame:
    if n_card <= 0:
        return _empty_frame()
    n_quiet = int(math.floor(n_card * _CARD_QUIET_FRACTION))
    n_loud = n_card - n_quiet
    parts = []
    if n_loud > 0:
        customer, device, merchant, amount, offset_s = _grouped_charges(
            rng,
            config,
            n_loud,
            max_offset,
            _CARD_GROUP,
            12,
            0.5,
            4.0,
            n_popular,
            n_clean_merchants,
            popular_fraction=0.30,
            shared_fraction=0.08,
        )
        parts.append(
            _frame_from_arrays(customer, device, merchant, amount, offset_s, 1, FRAUD_CARD)
        )
    if n_quiet > 0:
        customer, device, merchant, amount, offset_s = _grouped_charges(
            rng,
            config,
            n_quiet,
            max_offset,
            4,
            40,
            8.0,
            60.0,
            n_popular,
            n_clean_merchants,
            popular_fraction=0.80,
            shared_fraction=0.20,
        )
        parts.append(
            _frame_from_arrays(customer, device, merchant, amount, offset_s, 1, FRAUD_CARD)
        )
    if not parts:
        return _empty_frame()
    return pd.concat(parts, ignore_index=True)


def _ato_frame(
    rng: np.random.Generator,
    week: int,
    n_ato: int,
    customer: np.ndarray,
    offset: np.ndarray,
    max_offset: int,
    cust_log_amount: np.ndarray,
    n_popular: int,
) -> pd.DataFrame:
    if n_ato <= 0:
        return _empty_frame()
    n_legit = int(customer.shape[0])
    anchor = rng.choice(n_legit, size=n_ato, replace=False)
    legit_off = offset[anchor].astype(np.int64)
    new_legit = np.minimum(legit_off, max_offset - 2 * 3600 - 1)
    new_legit = np.maximum(new_legit, 0)
    ato_off = new_legit + 2 * 3600
    offset[anchor] = new_legit
    cust = customer[anchor]
    device = (_ATO_DEVICE_BASE + week * 1_000_000 + np.arange(n_ato, dtype=np.int64)).astype(
        np.int32
    )
    merchant = rng.integers(0, n_popular, size=n_ato, dtype=np.int32)
    amount = rng.uniform(400.0, 1400.0, size=n_ato)
    n_quiet = int(math.floor(n_ato * _ATO_QUIET_FRACTION))
    if n_quiet > 0:
        # Known device and a normal ticket. These sit on top of legitimate spend.
        device[:n_quiet] = cust[:n_quiet]
        quiet_amount = np.exp(cust_log_amount[cust[:n_quiet]]) * rng.uniform(0.8, 1.8, size=n_quiet)
        amount[:n_quiet] = np.clip(quiet_amount, 0.5, 2000.0)
    return _frame_from_arrays(cust, device, merchant, amount, ato_off, 1, FRAUD_ATO)


def _mule_frame(
    rng: np.random.Generator,
    config: SimConfig,
    n_mule: int,
    max_offset: int,
    cust_log_amount: np.ndarray,
    mule_ids: np.ndarray,
    n_popular: int,
) -> pd.DataFrame:
    if n_mule <= 0:
        return _empty_frame()
    customer = rng.integers(0, config.n_customers, size=n_mule, dtype=np.int32)
    n_popular_rows = int(math.floor(n_mule * _MULE_ON_POPULAR_FRACTION))
    merchant = np.empty(n_mule, dtype=np.int32)
    if n_popular_rows > 0:
        merchant[:n_popular_rows] = rng.integers(0, n_popular, size=n_popular_rows, dtype=np.int32)
    n_rest = n_mule - n_popular_rows
    if n_rest > 0:
        merchant[n_popular_rows:] = rng.choice(mule_ids, size=n_rest, replace=True)
    device = customer.copy()
    n_shared = int(math.floor(0.20 * n_mule))
    if n_shared > 0:
        shared_at = rng.choice(n_mule, size=n_shared, replace=False)
        device[shared_at] = (
            _SHARED_DEVICE_BASE + rng.integers(0, _n_shared_devices(config), size=n_shared)
        ).astype(np.int32)
    offset_s = rng.integers(0, max_offset, size=n_mule, dtype=np.int64)
    amount = np.exp(cust_log_amount[customer]) * rng.uniform(0.7, 1.4, size=n_mule)
    amount = np.clip(amount, 0.5, 2000.0)
    return _frame_from_arrays(
        customer, device, merchant, amount, offset_s, 1, FRAUD_MULE
    )


def _farm_frame(
    rng: np.random.Generator,
    config: SimConfig,
    n_farm: int,
    max_offset: int,
    cust_log_amount: np.ndarray,
    n_popular: int,
    n_clean_merchants: int,
) -> pd.DataFrame:
    if n_farm <= 0:
        return _empty_frame()
    replace = n_farm > config.n_customers
    customer = rng.choice(config.n_customers, size=n_farm, replace=replace).astype(np.int32)
    device = (_FARM_DEVICE_BASE + (np.arange(n_farm) % config.n_farm_devices)).astype(np.int32)
    merchant = rng.integers(0, n_clean_merchants, size=n_farm, dtype=np.int32)
    n_own = int(math.floor(n_farm * _FARM_ON_OWN_DEVICE_FRACTION))
    if n_own > 0:
        device[:n_own] = customer[:n_own]
        merchant[:n_own] = rng.integers(0, n_popular, size=n_own, dtype=np.int32)
    offset_s = np.sort(rng.integers(0, max_offset, size=n_farm, dtype=np.int64))
    amount = np.exp(cust_log_amount[customer]) * rng.uniform(0.8, 1.5, size=n_farm)
    amount = np.clip(amount, 0.5, 2000.0)
    return _frame_from_arrays(
        customer, device, merchant, amount, offset_s, 1, FRAUD_FARM
    )


def _paint_bursts(
    rng: np.random.Generator,
    customer: np.ndarray,
    device: np.ndarray,
    merchant: np.ndarray,
    amount: np.ndarray,
    offset: np.ndarray,
    row_idx: np.ndarray,
    group_size: int,
    spacing_s: int,
    amount_low: float,
    amount_high: float,
    max_offset: int,
) -> None:
    n = int(row_idx.size) - (int(row_idx.size) % group_size)
    if n < group_size:
        return
    groups = np.sort(row_idx[:n]).reshape(-1, group_size)
    lead = groups[:, 0]
    cust = customer[lead]
    dev = device[lead]
    merch = merchant[lead]
    start = np.minimum(offset[lead].astype(np.int64), max_offset - spacing_s * group_size - 1)
    start = np.maximum(start, 0)
    n_groups = groups.shape[0]
    burst_amount = rng.uniform(amount_low, amount_high, size=(n_groups, group_size))
    for k in range(group_size):
        rows = groups[:, k]
        customer[rows] = cust
        device[rows] = dev
        merchant[rows] = merch
        offset[rows] = start + k * spacing_s
        amount[rows] = burst_amount[:, k]


def _swap_labels(rng: np.random.Generator, week_df: pd.DataFrame) -> None:
    """Withhold some fraud labels and mark the same number of ordinary rows fraud.

    The withheld rows keep their fraud_type so a delayed chargeback is still
    visible in the generator. Disputes stay fraud_type 0. The fraud count does
    not change.
    """
    is_fraud = week_df["is_fraud"].to_numpy().copy()
    fraud_type = week_df["fraud_type"].to_numpy().copy()
    fraud_idx = np.flatnonzero(is_fraud == 1)
    legit_idx = np.flatnonzero(is_fraud == 0)
    n_swap = int(math.floor(fraud_idx.size * _LABEL_SWAP_FRACTION))
    if n_swap < 1 or n_swap >= legit_idx.size:
        return
    drop = rng.choice(fraud_idx, size=n_swap, replace=False)
    add = rng.choice(legit_idx, size=n_swap, replace=False)
    is_fraud[drop] = 0
    is_fraud[add] = 1
    fraud_type[add] = FRAUD_LEGIT
    week_df["is_fraud"] = is_fraud
    week_df["fraud_type"] = fraud_type


def _simulate_week(
    rng: np.random.Generator,
    config: SimConfig,
    week: int,
    cust_log_amount: np.ndarray,
    mule_ids: np.ndarray,
) -> pd.DataFrame:
    mix = allocate_mix(config.fraud_per_week, shifted=(week == SHIFT_WEEK))
    n_fraud = sum(mix.values())
    n_legit = config.n_per_week - n_fraud
    # Leave half a minute so a later unique-microsecond tweak cannot cross the week.
    max_offset = _WEEK_SECONDS - 30
    n_clean = config.n_merchants - config.n_mule_merchants
    n_popular = _n_popular(n_clean)
    n_shared = _n_shared_devices(config)

    customer = rng.integers(0, config.n_customers, size=n_legit, dtype=np.int32)
    offset = rng.integers(0, max_offset, size=n_legit, dtype=np.int64)
    merchant = rng.integers(0, n_clean, size=n_legit, dtype=np.int32)
    device = customer.copy()
    amount = rng.lognormal(mean=cust_log_amount[customer], sigma=0.40).astype(np.float64)
    amount = np.clip(amount, 0.5, 2000.0)

    # Disjoint slices so later overlays do not erase the mule and farm traffic.
    perm = rng.permutation(n_legit)
    cursor = 0

    def take(share: float) -> np.ndarray:
        nonlocal cursor
        n = int(round(share * n_legit))
        n = min(n, n_legit - cursor)
        chosen = perm[cursor : cursor + n]
        cursor += n
        return chosen

    popular_idx = take(0.15)
    if popular_idx.size:
        merchant[popular_idx] = rng.integers(0, n_popular, size=popular_idx.size, dtype=np.int32)
    # A few hundred legitimate rows on mule merchants and farm devices, not tens of thousands.
    # That keeps those entities enriched for fraud without making the rule perfect.
    mule_idx = take(0.0005)
    if mule_idx.size:
        merchant[mule_idx] = rng.choice(mule_ids, size=mule_idx.size, replace=True)
    farm_idx = take(0.0005)
    if farm_idx.size:
        device[farm_idx] = (
            _FARM_DEVICE_BASE + rng.integers(0, config.n_farm_devices, size=farm_idx.size)
        ).astype(np.int32)
    shared_idx = take(0.003)
    if shared_idx.size:
        device[shared_idx] = (
            _SHARED_DEVICE_BASE + rng.integers(0, n_shared, size=shared_idx.size)
        ).astype(np.int32)
    new_device_idx = take(0.001)
    if new_device_idx.size:
        device[new_device_idx] = (
            _LEGIT_NEW_DEVICE_BASE + week * 1_000_000 + np.arange(new_device_idx.size)
        ).astype(np.int32)
        n_new_high = max(1, new_device_idx.size // 8)
        amount[new_device_idx[:n_new_high]] = rng.uniform(400.0, 1400.0, size=n_new_high)
    bill_idx = take(0.04)
    if bill_idx.size:
        amount[bill_idx] = rng.uniform(280.0, 1000.0, size=bill_idx.size)
    # Ordinary bursts, then a thin set of tiny rapid charges that resemble card testing.
    _paint_bursts(
        rng, customer, device, merchant, amount, offset, take(0.002), 4, 40, 8.0, 45.0, max_offset
    )
    _paint_bursts(
        rng, customer, device, merchant, amount, offset, take(0.0004), 6, 12, 0.5, 4.0, max_offset
    )

    ato = _ato_frame(
        rng, week, mix["ato"], customer, offset, max_offset, cust_log_amount, n_popular
    )
    legit = _frame_from_arrays(customer, device, merchant, amount, offset, 0, FRAUD_LEGIT)
    parts = [
        legit,
        _card_test_frame(rng, config, mix["card_test"], max_offset, n_popular, n_clean),
        ato,
        _mule_frame(rng, config, mix["mule"], max_offset, cust_log_amount, mule_ids, n_popular),
        _farm_frame(
            rng, config, mix["device_farm"], max_offset, cust_log_amount, n_popular, n_clean
        ),
    ]
    week_df = pd.concat(parts, ignore_index=True)
    if len(week_df) != config.n_per_week:
        raise RuntimeError(f"week {week} has {len(week_df)} rows, expected {config.n_per_week}")
    _swap_labels(rng, week_df)
    if int(week_df["is_fraud"].sum()) != config.fraud_per_week:
        raise RuntimeError(f"week {week} fraud count drifted")

    week_start = _BASE + np.timedelta64((week - 1) * 7, "D")
    week_df["event_ts"] = week_start + week_df["offset_s"].to_numpy().astype("timedelta64[s]")
    week_df["week"] = np.int8(week)
    if week == SHIFT_WEEK:
        week_df["acq_region"] = np.zeros(len(week_df), dtype=np.int8)
    else:
        week_df["acq_region"] = (np.arange(len(week_df), dtype=np.int32) % 5).astype(np.int8)
    return week_df.drop(columns=["offset_s"])


def _derived_week(event_ts: pd.Series) -> np.ndarray:
    delta_days = (event_ts.to_numpy(dtype="datetime64[ns]") - _BASE) / np.timedelta64(1, "D")
    return (np.floor(delta_days / 7).astype(np.int16) + 1).astype(np.int8)


def simulate(config: SimConfig) -> tuple[pd.DataFrame, np.ndarray]:
    validate_config(config)
    rng = np.random.default_rng(config.seed)
    cust_log_amount = rng.normal(np.log(40.0), 0.25, size=config.n_customers)
    mule_ids = mule_merchant_ids(config)
    frames = [
        _simulate_week(rng, config, week, cust_log_amount, mule_ids)
        for week in range(1, config.n_weeks + 1)
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["event_ts", "customer_id"], kind="mergesort").reset_index(drop=True)
    df["txn_id"] = np.arange(len(df), dtype=np.int64)
    # Unique timestamps preserve the sorted order and keep RANGE windows from
    # treating same-second peers as a single point.
    df["event_ts"] = df["event_ts"] + pd.to_timedelta(df["txn_id"].to_numpy(), unit="us")
    derived = _derived_week(df["event_ts"])
    if not np.array_equal(derived, df["week"].to_numpy()):
        raise RuntimeError("a timestamp crossed a week boundary")
    if len(df) != config.n_transactions:
        raise RuntimeError("unexpected transaction count")
    if int(df["is_fraud"].sum()) != config.fraud_per_week * config.n_weeks:
        raise RuntimeError("unexpected fraud count")
    df["customer_id"] = df["customer_id"].astype(np.int32)
    df["card_id"] = df["card_id"].astype(np.int32)
    df["device_id"] = df["device_id"].astype(np.int32)
    df["merchant_id"] = df["merchant_id"].astype(np.int32)
    df["amount"] = df["amount"].astype(np.float64)
    df["is_fraud"] = df["is_fraud"].astype(np.int8)
    df["fraud_type"] = df["fraud_type"].astype(np.int8)
    df["acq_region"] = df["acq_region"].astype(np.int8)
    df["week"] = df["week"].astype(np.int8)
    column_order = [
        "txn_id",
        "event_ts",
        "week",
        "customer_id",
        "card_id",
        "device_id",
        "merchant_id",
        "amount",
        "is_fraud",
        "fraud_type",
        "acq_region",
    ]
    return df[column_order], mule_ids


def write_transactions(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checksum_path = path.parent / f"{path.name}.sha256"
    checksum_path.write_text(digest + "\n", encoding="utf-8")
    return digest


def verify_transactions(path: Path, expected_rows: int) -> str:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    recorded = (path.parent / f"{path.name}.sha256").read_text(encoding="utf-8").strip()
    if digest != recorded:
        raise ValueError("checksum mismatch for cached transactions")
    con = duckdb.connect()
    try:
        found = con.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(path)]).fetchone()
    finally:
        con.close()
    if found is None or int(found[0]) != int(expected_rows):
        raise ValueError(f"row count {found} != {expected_rows}")
    return digest
