"""Discover, validate, and describe deployable model artifacts."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from config import (
    CONFIG_SCHEMA_VERSION,
    FEATURE_NAMES,
    FEATURE_VERSION,
    RESAMPLING_METHOD,
    SUPPORTED_INPUT_SAMPLING_RATES,
    TARGET_SAMPLING_RATE,
    WINDOW_SIZE,
    preprocessing_config,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

EXPECTED_CLASSES = {
    "binary": ["Faulty", "Healthy"],
    "multiclass": ["Ball", "Inner_Race", "Normal", "Outer_Race"],
}


class ArtifactCompatibilityError(ValueError):
    """Raised when a saved artifact cannot satisfy the inference contract."""


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_identity():
    """Return stable identities for the manifest and processed feature table."""
    paths = {
        "manifest": ROOT / "docs/data_manifest.csv",
        "features_2048": ROOT / "data/processed/cwru_features.csv",
    }
    return {
        name: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
        for name, path in paths.items() if path.exists()
    }


def _model_family(artifact):
    model = artifact.get("model")
    try:
        class_name = type(model.named_steps["model"]).__name__
        return {
            "RandomForestClassifier": "rf",
            "SVC": "svm",
            "XGBClassifier": "xgboost",
        }.get(class_name, class_name)
    except (AttributeError, KeyError) as error:
        raise ArtifactCompatibilityError("artifact model must contain a 'model' pipeline step") from error


def _legacy_defaults():
    return {
        "artifact_schema_version": 0,
        "feature_version": FEATURE_VERSION,
        "preprocessing": preprocessing_config(),
        "window_size": WINDOW_SIZE,
        "target_sampling_rate": TARGET_SAMPLING_RATE,
        "supported_input_sampling_rates": list(SUPPORTED_INPUT_SAMPLING_RATES),
        "resampling_method": RESAMPLING_METHOD,
    }


def describe_artifact(path, validate=True):
    """Load an artifact and return its registry metadata without changing it."""
    path = Path(path)
    try:
        artifact = joblib.load(path)
    except Exception as error:
        raise ArtifactCompatibilityError(f"could not load artifact: {error}") from error

    task = artifact.get("task")
    if task not in EXPECTED_CLASSES:
        raise ArtifactCompatibilityError(f"unsupported or missing task: {task!r}")

    classes = [str(value) for value in artifact.get("classes", [])]
    features = [str(value) for value in artifact.get("features", [])]
    if validate and classes != EXPECTED_CLASSES[task]:
        raise ArtifactCompatibilityError(
            f"{path.name}: classes {classes!r} do not match {EXPECTED_CLASSES[task]!r}"
        )
    if validate and features != FEATURE_NAMES:
        raise ArtifactCompatibilityError(f"{path.name}: feature names/order do not match the approved 25-feature contract")

    defaults = _legacy_defaults()
    preprocessing = artifact.get("preprocessing", defaults["preprocessing"])
    window_size = int(artifact.get("window_size", defaults["window_size"]))
    target_rate = int(artifact.get("target_sampling_rate", defaults["target_sampling_rate"]))
    supported_rates = [int(value) for value in artifact.get(
        "supported_input_sampling_rates", defaults["supported_input_sampling_rates"]
    )]
    feature_version = artifact.get("feature_version", defaults["feature_version"])
    if validate:
        if window_size != WINDOW_SIZE or target_rate != TARGET_SAMPLING_RATE:
            raise ArtifactCompatibilityError(f"{path.name}: window or target sampling rate is incompatible")
        if supported_rates != list(SUPPORTED_INPUT_SAMPLING_RATES):
            raise ArtifactCompatibilityError(f"{path.name}: supported input rates are incompatible")
        if feature_version != FEATURE_VERSION:
            raise ArtifactCompatibilityError(f"{path.name}: feature version {feature_version!r} is unsupported")

    model_family = artifact.get("model_family", _model_family(artifact))
    validation = artifact.get("validation")
    if validation is None:
        validation = {
            "selection_scheme": "leave_one_load_out",
            "selection_metric": "macro_f1",
            "cv_macro_f1": artifact.get("cv_macro_f1", {}),
        }
    return {
        "artifact": str(path.relative_to(ROOT)),
        "artifact_schema_version": int(artifact.get("artifact_schema_version", 0)),
        "task": task,
        "model_family": model_family,
        "classes": classes,
        "features": features,
        "feature_count": len(features),
        "feature_version": feature_version,
        "window_size": window_size,
        "target_sampling_rate": target_rate,
        "supported_input_sampling_rates": supported_rates,
        "preprocessing": preprocessing,
        "selection": artifact.get("selection"),
        "validation": validation,
        "dataset_identity": artifact.get("dataset_identity", dataset_identity()),
        "legacy_artifact": "preprocessing" not in artifact,
        "created_at_utc": artifact.get("created_at_utc"),
        "available": True,
    }


def discover_registry(models_dir=MODELS_DIR, validate=True):
    """Return compatible artifact metadata and skip incompatible artifacts."""
    entries = []
    errors = []
    seen = set()
    for path in sorted(Path(models_dir).glob("*_model.joblib")):
        try:
            entry = describe_artifact(path, validate=validate)
            key = (entry["task"], entry["model_family"])
            if key in seen:
                continue
            seen.add(key)
            entries.append(entry)
        except ArtifactCompatibilityError as error:
            errors.append({"artifact": str(path.relative_to(ROOT)), "error": str(error)})
    return entries, errors


def write_registry(output=RESULTS_DIR / "model_registry.json"):
    """Write a machine-readable registry for the dashboard and tooling."""
    entries, errors = discover_registry()
    payload = {
        "registry_schema_version": CONFIG_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "feature_version": FEATURE_VERSION,
            "features": FEATURE_NAMES,
            "preprocessing": preprocessing_config(),
        },
        "models": entries,
        "rejected_artifacts": errors,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    payload = write_registry()
    print(f"Registered {len(payload['models'])} compatible model(s); rejected {len(payload['rejected_artifacts'])}.")
