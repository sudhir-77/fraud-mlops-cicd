"""Shared constants. The test weeks never enter training, rule tuning, or threshold selection."""

from __future__ import annotations

from dataclasses import dataclass


SEED = 42
N_WEEKS = 12
SHIFT_WEEK = 11
TRAIN_WEEKS = [1, 2, 3, 4, 5, 6, 7, 8]
VAL_WEEKS = [9]
TEST_WEEKS = [10, 11, 12]

COST_FP = 5.0
COST_FN = 100.0
PSI_ALERT = 0.2
ALERT_RATE = 0.01

FRAUD_TYPE_NAMES = ["legit", "card_test", "ato", "mule", "device_farm"]
FRAUD_LEGIT = 0
FRAUD_CARD = 1
FRAUD_ATO = 2
FRAUD_MULE = 3
FRAUD_FARM = 4

# Scored columns. acq_region is monitored for drift and is not a model input:
# week 11 forces every transaction into region 0, and including it would move
# the score for a routing change that is independent of the fraud label.
MODEL_FEATURES = [
    "log_amount",
    "amount",
    "cust_txn_count_1h",
    "cust_txn_count_24h",
    "cust_amount_24h",
    "cust_prior_mean_amount",
    "amount_over_prior_mean",
    "cust_txn_count_7d",
    "card_txn_count_1h",
    "device_txn_count_24h",
    "device_prior_customers",
    "merchant_txn_count_7d",
    "merchant_fraud_rate_7d",
    "merchant_mean_amount_7d",
    "is_new_device",
]

MONITOR_FEATURES = MODEL_FEATURES + ["acq_region"]

MODEL_NAME = "fraud_payment_scorer"
PROJECT_SLUG = "fraud-risk-mlops"


@dataclass(frozen=True)
class SimConfig:
    n_per_week: int
    n_customers: int
    n_merchants: int
    n_mule_merchants: int
    n_farm_devices: int
    fraud_per_week: int
    n_weeks: int = N_WEEKS
    seed: int = SEED

    @property
    def n_transactions(self) -> int:
        return self.n_per_week * self.n_weeks


@dataclass(frozen=True)
class TrainConfig:
    num_boost_round: int
    num_leaves: int
    min_child_samples: int
    learning_rate: float
    early_stopping_rounds: int
    load_seconds: int
    shap_rows: int
    load_rps: int = 50


def full_config() -> SimConfig:
    # 250_000 * 12 = 3_000_000, fraud_per_week / n_per_week = 0.012.
    return SimConfig(
        n_per_week=250_000,
        n_customers=150_000,
        n_merchants=4_000,
        n_mule_merchants=80,
        n_farm_devices=40,
        fraud_per_week=3_000,
    )


def quick_config() -> SimConfig:
    return SimConfig(
        n_per_week=3_000,
        n_customers=1_200,
        n_merchants=60,
        n_mule_merchants=6,
        n_farm_devices=4,
        fraud_per_week=36,
    )


def train_config_for(quick: bool) -> TrainConfig:
    if quick:
        return TrainConfig(
            num_boost_round=25,
            num_leaves=8,
            min_child_samples=5,
            learning_rate=0.1,
            early_stopping_rounds=8,
            load_seconds=1,
            shap_rows=80,
        )
    return TrainConfig(
        num_boost_round=80,
        num_leaves=15,
        min_child_samples=80,
        learning_rate=0.08,
        early_stopping_rounds=15,
        load_seconds=60,
        shap_rows=800,
    )
