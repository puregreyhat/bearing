#!/usr/bin/env python3
"""Export a deterministic browser demo from the canonical deployment artifacts.

The static site is an explanatory demo, not the supported SRS dashboard. Streamlit
and ``src/inference.py`` remain the source of truth for production-like inference.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import decimate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import extract_features
from config import FEATURE_NAMES, FEATURE_VERSION, preprocessing_config

SAMPLES = {"healthy": 97, "inner": 105, "ball": 118, "outer": 130}


def signal(file_id):
    data = loadmat(ROOT / "data/raw" / f"{file_id}.mat")
    key = f"X{file_id:03d}_DE_time"
    x = data[key].squeeze()
    return decimate(x, 4, ftype="fir", zero_phase=True) if file_id < 101 else x


def forest(artifact):
    model = artifact["model"].named_steps["model"]
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        values = tree.value[:, 0, :]
        totals = values.sum(axis=1, keepdims=True)
        values = np.divide(values, totals, out=np.zeros_like(values), where=totals != 0)
        trees.append({
            "l": tree.children_left.tolist(), "r": tree.children_right.tolist(),
            "f": tree.feature.tolist(), "t": np.round(tree.threshold, 10).tolist(),
            "v": np.round(values, 10).tolist(),
        })
    return {"classes": artifact["classes"].tolist(), "features": artifact["features"], "trees": trees}


def main():
    artifacts = {task: joblib.load(ROOT / "models" / f"{task}_model.joblib") for task in ("binary", "multiclass")}
    exported_samples = {}
    for label, file_id in SAMPLES.items():
        window = signal(file_id)[:2048]
        features = extract_features(window, 12000)
        row = pd.DataFrame([[features[name] for name in artifacts["binary"]["features"]]], columns=artifacts["binary"]["features"])
        exported_samples[label] = {
            "file": file_id,
            "wave": np.round(window[::4], 7).tolist(),
            "spectrum": np.round(np.abs(np.fft.rfft(window * np.hanning(len(window))))[::4], 6).tolist(),
            "features": {name: round(float(value), 6) for name, value in features.items()},
            "binary": dict(zip(artifacts["binary"]["classes"], np.round(artifacts["binary"]["model"].predict_proba(row)[0], 6))),
            "multiclass": dict(zip(artifacts["multiclass"]["classes"], np.round(artifacts["multiclass"]["model"].predict_proba(row)[0], 6))),
        }
    payload = {
        "contract": {
            "surface": "generated_browser_demo",
            "source_of_truth": "src/inference.py and the Streamlit dashboard",
            "model_families": ["rf"],
            "feature_version": FEATURE_VERSION,
            "features": FEATURE_NAMES,
            "preprocessing": preprocessing_config(),
            "limitations": [
                "Embedded samples and Random Forests only",
                "No arbitrary MAT/NPY upload or model selection",
                "Use Streamlit for the supported SRS workflow",
            ],
        },
        "samples": exported_samples,
        "models": {task: forest(artifact) for task, artifact in artifacts.items()},
    }
    output = ROOT / "website/dist/site-data.js"
    output.write_text("window.SITE_DATA=" + json.dumps(payload, separators=(",", ":")) + ";")
    print(f"Wrote {output} ({output.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
