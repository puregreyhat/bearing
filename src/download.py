#!/usr/bin/env python3
"""Download the selected real CWRU recordings and write their inspected manifest."""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
SOURCE = "https://engineering.case.edu/sites/default/files"
LOAD_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}


def registry():
    rows = []
    for load, file_id in enumerate(range(97, 101)):
        rows.append((file_id, "Normal", 0.0, load, 48000))
    groups = {
        ("Inner_Race", 0.007): range(105, 109),
        ("Ball", 0.007): range(118, 122),
        ("Outer_Race", 0.007): range(130, 134),
        ("Inner_Race", 0.014): range(169, 173),
        ("Ball", 0.014): range(185, 189),
        ("Outer_Race", 0.014): range(197, 201),
        ("Inner_Race", 0.021): range(209, 213),
        ("Ball", 0.021): range(222, 226),
        ("Outer_Race", 0.021): range(234, 238),
    }
    for (fault_type, diameter), ids in groups.items():
        rows.extend((file_id, fault_type, diameter, load, 12000) for load, file_id in enumerate(ids))
    return rows


def de_key(data, file_id):
    keys = [key for key in data if key.endswith("_DE_time")]
    expected = f"X{file_id:03d}_DE_time"
    if expected in keys:
        return expected
    if len(keys) == 1:
        return keys[0]
    raise ValueError(f"could not select channel for {file_id}; found {keys}")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = []
    for file_id, fault_type, diameter, load, source_rate in registry():
        url = f"{SOURCE}/{file_id}.mat"
        path = RAW / f"{file_id}.mat"
        if not path.exists():
            print(f"Downloading {file_id}.mat")
            urlretrieve(url, path)
        data = loadmat(path)
        signal_key = de_key(data, file_id)
        rpm_keys = [key for key in data if key.endswith("RPM")]
        measured_rpm = int(round(float(data[rpm_keys[0]].squeeze()))) if rpm_keys else LOAD_RPM[load]
        manifest.append({
            "file_id": str(file_id), "filename": path.name,
            "condition": "Healthy" if fault_type == "Normal" else "Faulty",
            "fault_type": fault_type, "fault_diameter": diameter,
            "load_hp": load, "rpm": measured_rpm,
            "sampling_rate": 12000,
            "channel": signal_key, "n_samples": int(data[signal_key].size),
            "source_url": url,
            "source_sampling_rate": source_rate,
            "processing": "decimate_by_4" if source_rate == 48000 else "none",
        })
        print(file_id, sorted(key for key in data if not key.startswith("__")))
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    pd.DataFrame(manifest).to_csv(docs / "data_manifest.csv", index=False)
    print(f"Wrote {len(manifest)} rows to {docs / 'data_manifest.csv'}")


if __name__ == "__main__":
    main()
