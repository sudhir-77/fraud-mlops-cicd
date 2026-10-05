"""FastAPI scoring. The score is LightGBM's probability; reason codes are TreeSHAP."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class ReasonCode(BaseModel):
    feature: str
    shap: float
    direction: str


class ScoreRequest(BaseModel):
    txn_id: int
    features: dict[str, float]


class ScoreResponse(BaseModel):
    txn_id: int
    score: float
    decision: str
    reason_codes: list[ReasonCode] = Field(default_factory=list)


def top_reason_codes(contrib: np.ndarray, names: list[str], k: int = 3) -> list[dict]:
    """`contrib` includes the trailing bias term from LightGBM pred_contrib."""
    values = np.asarray(contrib, dtype=np.float64)
    feature_values = values[:-1]
    if feature_values.size != len(names):
        raise ValueError("contribution length does not match feature names")
    k = min(k, feature_values.size)
    if k == 0:
        return []
    pick = np.argpartition(-np.abs(feature_values), k - 1)[:k]
    pick = pick[np.argsort(-np.abs(feature_values[pick]))]
    reasons = []
    for index in pick:
        shap_value = float(feature_values[index])
        if shap_value > 0:
            direction = "increases_risk"
        elif shap_value < 0:
            direction = "decreases_risk"
        else:
            direction = "neutral"
        reasons.append(
            {"feature": names[int(index)], "shap": shap_value, "direction": direction}
        )
    return reasons


def create_app(
    booster: lgb.Booster,
    threshold: float,
    feature_names: list[str],
) -> FastAPI:
    app = FastAPI(title="fraud-risk-mlops", version="1.0.0")
    lock = threading.Lock()
    names = list(feature_names)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/score", response_model=ScoreResponse)
    def score(body: ScoreRequest) -> ScoreResponse:
        missing = [name for name in names if name not in body.features]
        if missing:
            raise HTTPException(status_code=400, detail=f"missing features: {missing}")
        row = np.empty((1, len(names)), dtype=np.float64)
        for i, name in enumerate(names):
            row[0, i] = body.features[name]
        if not np.isfinite(row).all():
            raise HTTPException(status_code=400, detail="features must be finite")
        with lock:
            probability = float(booster.predict(row, num_threads=1)[0])
            contrib = booster.predict(row, pred_contrib=True, num_threads=1)[0]
        reasons = top_reason_codes(contrib, names, k=3)
        decision = "review" if probability >= threshold else "approve"
        return ScoreResponse(
            txn_id=body.txn_id,
            score=probability,
            decision=decision,
            reason_codes=[ReasonCode(**item) for item in reasons],
        )

    return app


def create_app_from_env() -> FastAPI:
    root = Path(os.environ["FRAUD_PROJECT_ROOT"])
    spec = json.loads((root / "models" / "serving.json").read_text(encoding="utf-8"))
    model_path = root / spec["model_path"]
    booster = lgb.Booster(model_file=str(model_path))
    return create_app(booster, float(spec["threshold"]), list(spec["feature_names"]))
