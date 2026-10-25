"""Shared signal loading and model inference used by the dashboard and smoke checks."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import decimate

from config import SUPPORTED_INPUT_SAMPLING_RATES, TARGET_SAMPLING_RATE, WINDOW_SIZE
from contracts import InferenceResult
from features import extract_features
from model_registry import ArtifactCompatibilityError, describe_artifact, discover_registry

ROOT = Path(__file__).resolve().parents[1]


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
    if signal.size == 0:
        raise ValueError("signal is empty")
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if not np.isfinite(signal).all():
        raise ValueError("signal contains NaN or infinity")
    return signal


def prepare_windows(signal, sampling_rate):
    windows, _ = prepare_windows_with_metadata(signal, sampling_rate)
    return windows


def prepare_windows_with_metadata(signal, sampling_rate):
    """Normalize sampling rate and return windows plus processing metadata."""
    signal = np.asarray(signal, dtype=float).squeeze()
    original_count = len(signal)
    if sampling_rate == 48000:
        signal = decimate(signal, 4, ftype="fir", zero_phase=True)
    elif sampling_rate not in SUPPORTED_INPUT_SAMPLING_RATES:
        raise ValueError("sampling rate must be 12,000 or 48,000 Hz")
    count = len(signal) // WINDOW_SIZE
    if not count:
        raise ValueError(f"signal must contain at least {WINDOW_SIZE} samples after resampling")
    return signal[:count * WINDOW_SIZE].reshape(count, WINDOW_SIZE), {
        "source_sampling_rate": int(sampling_rate),
        "processed_sampling_rate": TARGET_SAMPLING_RATE,
        "source_sample_count": int(original_count),
        "processed_sample_count": int(len(signal)),
        "valid_window_count": int(count),
        "discarded_tail_samples": int(len(signal) - count * WINDOW_SIZE),
    }


def _selected_artifact(task, model_family=None):
    entries, errors = discover_registry()
    candidates = [entry for entry in entries if entry["task"] == task]
    if model_family is not None:
        candidates = [entry for entry in candidates if entry["model_family"] == model_family]
    if not candidates:
        detail = f" for model {model_family!r}" if model_family else ""
        rejected = f"; rejected artifacts: {errors}" if errors else ""
        raise ArtifactCompatibilityError(f"no compatible {task!r} artifact{detail}{rejected}")
    entry = candidates[0]
    return entry, joblib.load(ROOT / entry["artifact"])


def predict_selected(windows, task, model_family=None):
    """Predict one selected task/model and return window and recording outputs."""
    entry, artifact = _selected_artifact(task, model_family)
    frame = pd.DataFrame([extract_features(window, TARGET_SAMPLING_RATE) for window in windows])[artifact["features"]]
    window_probabilities = artifact["model"].predict_proba(frame)
    classes = [str(value) for value in artifact["classes"]]
    window_dicts = [dict(zip(classes, map(float, row))) for row in window_probabilities]
    probabilities = {
        label: float(window_probabilities[:, index].mean())
        for index, label in enumerate(classes)
    }
    predictions = [classes[index] for index in window_probabilities.argmax(axis=1)]
    return entry, artifact, probabilities, window_dicts, predictions, frame


def analyze_signal(signal, sampling_rate, task, model_family=None, filename=None):
    """Run the complete selected-task inference contract for one recording."""
    windows, processing = prepare_windows_with_metadata(signal, sampling_rate)
    entry, artifact, probabilities, window_probabilities, predictions, frame = predict_selected(
        windows, task, model_family
    )
    predicted_class = max(probabilities, key=probabilities.get)
    confidence = probabilities[predicted_class]
    health_index = 100 * probabilities["Healthy"] if "Healthy" in probabilities else None
    warnings = []
    if processing["discarded_tail_samples"]:
        warnings.append("Incomplete tail samples were discarded.")
    if sampling_rate in SUPPORTED_INPUT_SAMPLING_RATES and sampling_rate != TARGET_SAMPLING_RATE:
        warnings.append("The input was resampled to 12,000 Hz before feature extraction.")
    return InferenceResult(
        task=task,
        model_family=entry["model_family"],
        classes=[str(value) for value in artifact["classes"]],
        input_filename=filename,
        source_sampling_rate=processing["source_sampling_rate"],
        processed_sampling_rate=processing["processed_sampling_rate"],
        sample_count=processing["source_sample_count"],
        duration_seconds=processing["source_sample_count"] / sampling_rate,
        window_size=WINDOW_SIZE,
        valid_window_count=processing["valid_window_count"],
        discarded_tail_samples=processing["discarded_tail_samples"],
        probabilities=probabilities,
        window_probabilities=window_probabilities,
        window_predictions=predictions,
        predicted_class=predicted_class,
        confidence=confidence,
        health_index=health_index,
        first_window=[float(value) for value in windows[0]],
        first_window_features={name: float(value) for name, value in frame.iloc[0].to_dict().items()},
        warnings=warnings,
    )


def predict(windows):
    artifacts = {task: joblib.load(ROOT / "models" / f"{task}_model.joblib") for task in ("binary", "multiclass")}
    features = artifacts["binary"]["features"]
    frame = pd.DataFrame([extract_features(window, TARGET_SAMPLING_RATE) for window in windows])[features]
    output = {}
    for task, artifact in artifacts.items():
        probabilities = artifact["model"].predict_proba(frame).mean(axis=0)
        output[task] = dict(zip(artifact["classes"], probabilities))
    return output
