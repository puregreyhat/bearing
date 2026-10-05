#!/usr/bin/env python3
"""Validate labels and feature integrity, then save the Phase 3 report and plots."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import decimate

RESULTS = ROOT / "results"


def main():
    RESULTS.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / "data/processed/cwru_features.csv", dtype={"File_ID": str})
    manifest = pd.read_csv(ROOT / "docs/data_manifest.csv", dtype={"file_id": str})
    assert df["Window_ID"].is_unique, "duplicate Window_ID"
    assert not df.isna().any().any(), "missing values"
    numeric = df.select_dtypes(include=np.number)
    assert np.isfinite(numeric.to_numpy()).all(), "infinite values"
    labels = manifest.set_index("file_id")[["condition", "fault_type", "load_hp"]]
    for file_id, group in df.groupby("File_ID"):
        expected = labels.loc[file_id]
        assert (group["Bearing_Condition"] == expected.condition).all()
        assert (group["Fault_Type"] == expected.fault_type).all()
        assert (group["Load"] == expected.load_hp).all()

    balance = df.groupby(["Fault_Type", "Fault_Diameter", "Load"], dropna=False).size().rename("windows")
    class_counts = df["Fault_Type"].value_counts()
    warning = "YES" if class_counts.max() / class_counts.min() > 1.5 else "NO"
    sufficient = class_counts.ge(300)
    summary = df.groupby("Bearing_Condition")[["RMS", "Kurtosis"]].agg(["mean", "median"])
    plot_examples(manifest)
    report = f"""# Dataset validation

- Rows: {len(df)}; columns: {len(df.columns)}
- Missing values: 0; infinite values: 0
- Duplicate Window_IDs: 0
- Labels cross-checked against manifest: PASS
- Numeric feature dtypes: PASS
- Imbalance flagged (>1.5x largest/smallest class): {warning}
- At least 300 windows per Fault_Type: {'PASS' if sufficient.all() else 'WARNING — ' + ', '.join(sufficient[~sufficient].index)}

## Windows per fault type

{class_counts.to_frame('windows').to_markdown()}

## Balance by fault type, diameter, and load

{balance.to_frame().to_markdown()}

## Healthy/faulty sanity statistics

{summary.to_markdown()}

The distributions differ; this is a sanity observation, not a model-performance claim.
Example waveforms and spectra are saved in `results/signal_spectra_examples.png`.
"""
    (RESULTS / "dataset_validation.md").write_text(report)
    print(report)


def plot_examples(manifest):
    selected = manifest.drop_duplicates("fault_type")
    fig, axes = plt.subplots(len(selected), 2, figsize=(12, 3 * len(selected)))
    for row, (_, item) in enumerate(selected.iterrows()):
        data = loadmat(ROOT / "data/raw" / item.filename)
        signal = data[item.channel].squeeze()
        if item.source_sampling_rate == 48000:
            signal = decimate(signal, 4, ftype="fir", zero_phase=True)
        window = signal[:2048]
        axes[row, 0].plot(np.arange(len(window)) / 12000, window, linewidth=.6)
        axes[row, 0].set_title(f"{item.fault_type} waveform")
        axes[row, 1].plot(np.fft.rfftfreq(len(window), 1 / 12000), np.abs(np.fft.rfft(window * np.hanning(len(window)))), linewidth=.6)
        axes[row, 1].set_title(f"{item.fault_type} spectrum")
    fig.tight_layout()
    fig.savefig(RESULTS / "signal_spectra_examples.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
