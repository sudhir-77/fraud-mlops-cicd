"""LightGBM trainer.

The function accepts only the matrices the caller already filtered to train and
validation weeks. It has no test argument.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np

from src.config import SEED, TrainConfig


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    train_cfg: TrainConfig,
) -> lgb.Booster:
    y_train = np.asarray(y_train)
    positives = max(int((y_train == 1).sum()), 1)
    negatives = int((y_train == 0).sum())
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": train_cfg.learning_rate,
        "num_leaves": train_cfg.num_leaves,
        "min_child_samples": train_cfg.min_child_samples,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "scale_pos_weight": float(negatives / positives),
        "verbosity": -1,
        "seed": SEED,
        "bagging_seed": SEED,
        "feature_fraction_seed": SEED,
        "data_random_seed": SEED,
        # One thread plus deterministic=True. Several threads still reshuffle
        # histogram reductions, and the cost threshold then moves between runs.
        "num_threads": 1,
        "deterministic": True,
        "force_row_wise": True,
        "feature_pre_filter": False,
    }
    dtrain = lgb.Dataset(
        x_train,
        label=y_train,
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        x_val,
        label=y_val,
        feature_name=list(feature_names),
        reference=dtrain,
        free_raw_data=False,
    )
    return lgb.train(
        params,
        dtrain,
        num_boost_round=train_cfg.num_boost_round,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(train_cfg.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
