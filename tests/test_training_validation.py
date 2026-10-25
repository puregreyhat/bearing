from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_training_outputs_record_validation_and_dataset_identity():
    artifact = __import__("joblib").load(ROOT / "models" / "binary_rf_model.joblib")
    assert artifact["validation"]["selection_scheme"] == "leave_one_load_out"
    assert artifact["validation"]["selection_metric"] == "macro_f1"
    assert artifact["dataset_identity"]["features_2048"]


def test_processed_features_have_unique_file_groups_and_labels():
    frame = pd.read_csv(ROOT / "data/processed/cwru_features.csv")
    assert frame["File_ID"].nunique() == frame.groupby("File_ID").ngroups
    assert frame["Bearing_Condition"].notna().all()
    assert frame["File_ID"].nunique() >= 4

