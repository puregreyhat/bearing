"""Migrate legacy joblib artifacts to the Phase 1 metadata schema."""

import joblib
from datetime import datetime, timezone
from pathlib import Path

from config import (
    FEATURE_VERSION,
    SUPPORTED_INPUT_SAMPLING_RATES,
    TARGET_SAMPLING_RATE,
    WINDOW_SIZE,
    preprocessing_config,
)
from model_registry import MODELS_DIR, dataset_identity, describe_artifact


def migrate(path):
    path = Path(path)
    artifact = joblib.load(path)
    metadata = describe_artifact(path)
    artifact.update({
        "artifact_schema_version": 1,
        "model_family": metadata["model_family"],
        "feature_version": FEATURE_VERSION,
        "window_size": WINDOW_SIZE,
        "target_sampling_rate": TARGET_SAMPLING_RATE,
        "supported_input_sampling_rates": list(SUPPORTED_INPUT_SAMPLING_RATES),
        "resampling_method": preprocessing_config()["resampling_method"],
        "preprocessing": preprocessing_config(),
        "validation": metadata["validation"],
        "dataset_identity": dataset_identity(),
        "created_at_utc": artifact.get("created_at_utc", datetime.now(timezone.utc).isoformat()),
        "migrated_at_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_artifact": False,
    })
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(artifact, temporary)
    reloaded = joblib.load(temporary)
    if reloaded.get("artifact_schema_version") != 1:
        raise RuntimeError(f"migration verification failed for {path.name}")
    temporary.replace(path)
    return path


def main():
    paths = sorted(MODELS_DIR.glob("*_model.joblib"))
    for path in paths:
        migrate(path)
        print(f"Migrated {path}")


if __name__ == "__main__":
    main()
