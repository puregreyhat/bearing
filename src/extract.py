#!/usr/bin/env python3
"""Downsample, segment, and extract features from untouched CWRU .mat files."""

from pathlib import Path

import pandas as pd
from scipy.io import loadmat
from scipy.signal import decimate

from features import extract_features

ROOT = Path(__file__).resolve().parents[1]


def build(window_size):
    manifest = pd.read_csv(ROOT / "docs/data_manifest.csv", dtype={"file_id": str})
    rows = []
    for item in manifest.to_dict("records"):
        data = loadmat(ROOT / "data/raw" / item["filename"])
        signal = data[item["channel"]].squeeze()
        if item["source_sampling_rate"] == 48000:
            signal = decimate(signal, 4, ftype="fir", zero_phase=True)
        for index, start in enumerate(range(0, len(signal) - window_size + 1, window_size)):
            row = {
                "Window_ID": f"{item['file_id']}_{window_size}_{index:04d}",
                "Bearing_Condition": item["condition"], "Fault_Type": item["fault_type"],
                "Load": item["load_hp"],
                **extract_features(signal[start:start + window_size], 12000),
                "File_ID": item["file_id"], "Fault_Diameter": item["fault_diameter"],
                "RPM": item["rpm"], "Window_Size": window_size, "Sampling_Rate": 12000,
            }
            rows.append(row)
    required = ["Window_ID", "Bearing_Condition", "Fault_Type", "Load", "RMS", "Std", "Peak",
                "Kurtosis", "Crest_Factor", "FFT_Peak_Frequency"]
    frame = pd.DataFrame(rows)
    frame = frame[required + [column for column in frame if column not in required]]
    output = ROOT / "data/processed" / ("cwru_features.csv" if window_size == 2048 else f"cwru_features_{window_size}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} windows to {output}")


if __name__ == "__main__":
    for size in (2048, 4096):
        build(size)
