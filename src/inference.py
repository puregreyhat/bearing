"""Shared signal loading and model inference used by the dashboard and smoke checks."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import decimate

from features import extract_features

ROOT = Path(__file__).resolve().parents[1]
WINDOW_SIZE = 2048


def read_signal(source, filename, mat_key=None):
    suffix = Path(filename).suffix.lower()
    if suffix == ".npy":
        signal = np.load(source, allow_pickle=False)
    elif suffix == ".csv":
        signal = np.loadtxt(source, delimiter=",")
    elif suffix == ".mat":
        data = loadmat(source)
        keys = [key for key in data if key.endswith("_DE_time")]
        expected = f"X{int(Path(filename).stem):03d}_DE_time" if Path(filename).stem.isdigit() else None
        key = mat_key or (expected if expected in keys else keys[0] if len(keys) == 1 else None)
        if not key:
            raise ValueError(f"ambiguous Drive-End channel: {keys}")
        signal = data[key]
    else:
        raise ValueError("upload a .csv, .mat, or .npy signal")
    signal = np.asarray(signal, dtype=float).squeeze()
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if not np.isfinite(signal).all():
        raise ValueError("signal contains NaN or infinity")
    return signal


def prepare_windows(signal, sampling_rate):
    if sampling_rate == 48000:
        signal = decimate(signal, 4, ftype="fir", zero_phase=True)
    elif sampling_rate != 12000:
        raise ValueError("sampling rate must be 12,000 or 48,000 Hz")
    count = len(signal) // WINDOW_SIZE
    if not count:
        raise ValueError(f"signal must contain at least {WINDOW_SIZE} samples after resampling")
    return signal[:count * WINDOW_SIZE].reshape(count, WINDOW_SIZE)


def predict(windows):
    artifacts = {task: joblib.load(ROOT / "models" / f"{task}_model.joblib") for task in ("binary", "multiclass")}
    features = artifacts["binary"]["features"]
    frame = pd.DataFrame([extract_features(window, 12000) for window in windows])[features]
    output = {}
    for task, artifact in artifacts.items():
        probabilities = artifact["model"].predict_proba(frame).mean(axis=0)
        output[task] = dict(zip(artifact["classes"], probabilities))
    return output
