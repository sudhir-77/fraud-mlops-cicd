"""Register a LightGBM booster and promote it Staging then Production."""

from __future__ import annotations

import warnings
from pathlib import Path

import lightgbm as lgb
import mlflow
from mlflow.tracking import MlflowClient

from src.config import MODEL_NAME


def promote_model(
    booster: lgb.Booster,
    registry_dir: Path,
    run_metrics: dict[str, float],
    params: dict[str, float | int | str],
) -> dict[str, str]:
    registry_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(registry_dir.resolve().as_uri())
    mlflow.set_experiment("fraud-risk-mlops")
    with mlflow.start_run(run_name="lgbm-fraud"):
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics({key: float(value) for key, value in run_metrics.items()})
        info = mlflow.lightgbm.log_model(
            booster,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            pip_requirements=[
                "lightgbm==4.5.0",
                "numpy==1.26.4",
                "pandas==2.2.2",
                "scikit-learn==1.5.1",
            ],
            await_registration_for=60,
        )
        run_id = mlflow.active_run().info.run_id
    client = MlflowClient(tracking_uri=registry_dir.resolve().as_uri())
    version = getattr(info, "registered_model_version", None)
    if not version:
        found = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not found:
            raise RuntimeError("model registration did not produce a version")
        version = str(max(int(item.version) for item in found))
    version = str(version)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Staging",
            archive_existing_versions=False,
        )
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
    current = client.get_model_version(MODEL_NAME, version)
    return {
        "model_name": MODEL_NAME,
        "version": version,
        "stage": str(current.current_stage),
        "run_id": run_id,
    }
