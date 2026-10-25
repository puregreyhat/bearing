#!/usr/bin/env python3
"""Leakage-safe nested evaluation and final model fitting for both diagnosis tasks."""

import json
import hashlib
import os
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut,
                                     StratifiedGroupKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import (FEATURE_NAMES, FEATURE_VERSION, SUPPORTED_INPUT_SAMPLING_RATES,
                    TARGET_SAMPLING_RATE, WINDOW_SIZE, preprocessing_config)

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
EXCLUDED = {
    "File_ID", "Fault_Diameter", "RPM", "Load", "Window_Size", "Sampling_Rate",
    "Window_ID", "Bearing_Condition", "Fault_Type",
}


def model_specs(n_classes):
    return {
        "svm": (
            Pipeline([("scale", StandardScaler()), ("model", SVC(class_weight="balanced", probability=True, random_state=SEED))]),
            {"model__C": [1, 10], "model__gamma": ["scale", 0.01]},
        ),
        "rf": (
            Pipeline([("model", RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=SEED, n_jobs=-1))]),
            {"model__max_depth": [None, 12], "model__min_samples_leaf": [1, 3]},
        ),
        "xgboost": (
            Pipeline([("model", XGBClassifier(n_estimators=200, subsample=.9, colsample_bytree=.9,
                                               random_state=SEED, n_jobs=4,
                                               objective="binary:logistic" if n_classes == 2 else "multi:softprob",
                                               eval_metric="logloss" if n_classes == 2 else "mlogloss"))]),
            {"model__max_depth": [3, 5], "model__learning_rate": [.05, .1]},
        ),
    }


def splits(name, x, y, files, loads):
    if name == "stratified_group_4fold":
        splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
        yield from splitter.split(x, y, files)
    else:
        yield from LeaveOneGroupOut().split(x, y, loads)


def fold_metrics(y, pred, proba, classes, elapsed_ms):
    precision, recall, _, _ = precision_recall_fscore_support(y, pred, labels=np.arange(len(classes)), zero_division=0)
    result = {
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "accuracy_not_headline": accuracy_score(y, pred),
        "inference_ms_per_window": elapsed_ms,
    }
    for index, label in enumerate(classes):
        result[f"precision_{label}"] = precision[index]
        result[f"recall_{label}"] = recall[index]
    if len(classes) == 2:
        healthy = int(np.where(classes == "Healthy")[0][0])
        truth = (y == healthy).astype(int)
        result["pr_auc_Healthy"] = average_precision_score(truth, proba[:, healthy])
        result["roc_auc_Healthy"] = roc_auc_score(truth, proba[:, healthy])
    return result


def fit_grid(name, pipeline, grid, x, y, files):
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipeline, grid, scoring="f1_macro", cv=inner, n_jobs=1, refit=True)
    fit_args = {"groups": files}
    if name == "xgboost":
        fit_args["model__sample_weight"] = compute_sample_weight("balanced", y)
    return search.fit(x, y, **fit_args)


def modal_params(rows, task, model):
    values = [row["best_params"] for row in rows if row["task"] == task and row["model"] == model and row["scheme"] == "leave_one_load_out"]
    return json.loads(Counter(values).most_common(1)[0][0])


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_payload(pipeline, task, model_name, classes, features, means, params, selected):
    """Build the self-describing deployment artifact for one task/model."""
    return {
        "artifact_schema_version": 1,
        "artifact_role": "deployment_candidate",
        "deployment_selected": bool(selected),
        "model": pipeline,
        "model_family": model_name,
        "features": features,
        "feature_version": FEATURE_VERSION,
        "classes": classes,
        "task": task,
        "window_size": WINDOW_SIZE,
        "target_sampling_rate": TARGET_SAMPLING_RATE,
        "supported_input_sampling_rates": list(SUPPORTED_INPUT_SAMPLING_RATES),
        "resampling_method": preprocessing_config()["resampling_method"],
        "preprocessing": preprocessing_config(),
        "hyperparameters": params,
        "selection": "highest mean Leave-One-Load-Out macro-F1" if selected else "Leave-One-Load-Out candidate",
        "validation": {
            "selection_scheme": "leave_one_load_out",
            "selection_metric": "macro_f1",
            "cv_macro_f1": means.to_dict(),
        },
        "dataset_identity": {
            "manifest": file_sha256(ROOT / "docs/data_manifest.csv"),
            "features_2048": file_sha256(ROOT / "data/processed/cwru_features.csv"),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main():
    np.random.seed(SEED)
    results = ROOT / "results"
    models_dir = ROOT / "models"
    results.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / "data/processed/cwru_features.csv", dtype={"File_ID": str})
    features = [column for column in df if column not in EXCLUDED]
    assert features == FEATURE_NAMES, "feature order must match the shared inference contract"
    assert not EXCLUDED.intersection(features)
    assert all(pd.api.types.is_numeric_dtype(df[column]) for column in features)
    (results / "feature_list.json").write_text(json.dumps(features, indent=2))
    print(f"FINAL FEATURE LIST ({len(features)} signal features only):")
    print("\n".join(features), flush=True)

    x = df[features]
    files = df["File_ID"].to_numpy()
    loads = df["Load"].to_numpy()
    rows, predictions = [], []
    importance = []
    tasks = {"binary": df["Bearing_Condition"], "multiclass": df["Fault_Type"]}
    for task, labels in tasks.items():
        encoder = LabelEncoder().fit(labels)
        y = encoder.transform(labels)
        classes = encoder.classes_
        for scheme in ("stratified_group_4fold", "leave_one_load_out"):
            for fold, (train, test) in enumerate(splits(scheme, x, y, files, loads), 1):
                assert set(files[train]).isdisjoint(files[test])
                assert set(np.unique(y[test])) == set(range(len(classes))), "every test fold must contain every class"
                held_out_load = int(np.unique(loads[test])[0]) if scheme == "leave_one_load_out" else None
                print(f"{task} {scheme} fold={fold} test_files={sorted(set(files[test]))}", flush=True)
                for name, (pipeline, grid) in model_specs(len(classes)).items():
                    training_started = time.perf_counter()
                    search = fit_grid(name, pipeline, grid, x.iloc[train], y[train], files[train])
                    training_seconds = time.perf_counter() - training_started
                    started = time.perf_counter()
                    pred = search.predict(x.iloc[test])
                    proba = search.predict_proba(x.iloc[test])
                    elapsed = (time.perf_counter() - started) * 1000 / len(test)
                    metrics = fold_metrics(y[test], pred, proba, classes, elapsed)
                    rows.append({
                        "task": task, "scheme": scheme, "model": name, "fold": fold,
                        "held_out_load": held_out_load, "train_files": len(set(files[train])),
                        "test_files": len(set(files[test])), "train_windows": len(train), "test_windows": len(test),
                        "best_params": json.dumps(search.best_params_, sort_keys=True),
                        "training_time_seconds": training_seconds, **metrics,
                    })
                    for position, index in enumerate(test):
                        predictions.append({
                            "task": task, "scheme": scheme, "model": name, "fold": fold,
                            "Window_ID": df.iloc[index].Window_ID, "File_ID": files[index], "Load": loads[index],
                            "true": classes[y[index]], "predicted": classes[pred[position]],
                        })
                    if scheme == "leave_one_load_out" and name in {"rf", "xgboost"}:
                        perm = permutation_importance(search.best_estimator_, x.iloc[test], y[test],
                                                      scoring="f1_macro", n_repeats=3, random_state=SEED, n_jobs=1)
                        native = search.best_estimator_.named_steps["model"].feature_importances_
                        importance.extend({"task": task, "model": name, "fold": fold, "feature": feature,
                                           "native_importance": native[i], "permutation_importance": perm.importances_mean[i]}
                                          for i, feature in enumerate(features))
                    print(name, {key: round(value, 4) for key, value in metrics.items() if key != "accuracy_not_headline"}, flush=True)

    pd.DataFrame(rows).to_csv(results / "fold_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(results / "fold_predictions.csv", index=False)
    pd.DataFrame(importance).to_csv(results / "feature_importance_folds.csv", index=False)

    # Select by the primary Leave-One-Load-Out macro-F1, then refit on all real data for deployment.
    metrics = pd.DataFrame(rows)
    logo = metrics[metrics.scheme == "leave_one_load_out"]
    for task, labels in tasks.items():
        encoder = LabelEncoder().fit(labels)
        y = encoder.transform(labels)
        means = logo[logo.task == task].groupby("model").macro_f1.mean()
        winner = means.idxmax()
        for model_name in model_specs(len(encoder.classes_)):
            params = modal_params(rows, task, model_name)
            pipeline = model_specs(len(encoder.classes_))[model_name][0].set_params(**params)
            fit_args = {"model__sample_weight": compute_sample_weight("balanced", y)} if model_name == "xgboost" else {}
            pipeline.fit(x, y, **fit_args)
            artifact = artifact_payload(
                pipeline, task, model_name, encoder.classes_, features, means, params, model_name == winner
            )
            candidate_path = models_dir / f"{task}_{model_name}_model.joblib"
            joblib.dump(artifact, candidate_path)
            print(f"Saved {task} candidate: {model_name} {params}")
            if model_name == winner:
                joblib.dump(artifact, models_dir / f"{task}_model.joblib")
                print(f"Saved {task} deployment alias: {winner}")


if __name__ == "__main__":
    main()
