from pathlib import Path

import pytest

from model_registry import ArtifactCompatibilityError, describe_artifact, discover_registry


ROOT = Path(__file__).resolve().parents[1]


def test_registry_contains_compatible_candidate_models():
    entries, errors = discover_registry()
    non_xgb_errors = [error for error in errors if "xgboost" not in error["error"]]
    assert not non_xgb_errors
    assert {entry["task"] for entry in entries} == {"binary", "multiclass"}
    assert {entry["model_family"] for entry in entries} >= {"rf", "svm"}


def test_feature_mismatch_is_rejected(tmp_path):
    import joblib

    source = joblib.load(ROOT / "models" / "binary_rf_model.joblib")
    source["features"] = list(reversed(source["features"]))
    path = tmp_path / "bad_model.joblib"
    joblib.dump(source, path)
    with pytest.raises(ArtifactCompatibilityError, match="feature names/order"):
        describe_artifact(path)


def test_task_and_class_mapping_mismatch_is_rejected(tmp_path):
    import joblib

    source = joblib.load(ROOT / "models" / "binary_rf_model.joblib")
    source["classes"] = ["Normal", "Faulty"]
    path = tmp_path / "bad_classes.joblib"
    joblib.dump(source, path)
    with pytest.raises(ArtifactCompatibilityError, match="classes"):
        describe_artifact(path)

