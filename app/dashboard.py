#!/usr/bin/env python3
"""Streamlit dashboard for CWRU-trained bearing diagnosis."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from inference import predict, prepare_windows, read_signal

st.set_page_config(page_title="Bearing Health", layout="wide")
st.title("Bearing Fault Diagnosis")
st.caption("CWRU laboratory benchmark — not yet validated for ceiling-fan motors")

source = st.radio("Signal source", ["Upload", "CWRU sample"], horizontal=True)
signal = None
sampling_rate = 12000
try:
    if source == "Upload":
        upload = st.file_uploader("Upload a numeric signal", type=["csv", "mat", "npy"])
        sampling_rate = st.selectbox("Input sampling rate", [12000, 48000])
        if upload:
            signal = read_signal(upload, upload.name)
    else:
        manifest = pd.read_csv(ROOT / "docs/data_manifest.csv", dtype={"file_id": str})
        file_id = st.selectbox("Recording", manifest.file_id)
        item = manifest.loc[manifest.file_id == file_id].iloc[0]
        path = ROOT / "data/raw" / item.filename
        signal = read_signal(path, item.filename, item.channel)
        sampling_rate = int(item.source_sampling_rate)

    if signal is not None:
        windows = prepare_windows(signal, sampling_rate)
        if len(signal) % (2048 * (4 if sampling_rate == 48000 else 1)):
            st.info("Incomplete tail samples were ignored.")
        probabilities = predict(windows)
        binary = probabilities["binary"]
        multiclass = probabilities["multiclass"]
        condition = max(binary, key=binary.get)
        fault = max(multiclass, key=multiclass.get)
        health = 100 * binary["Healthy"]

        a, b, c, d = st.columns(4)
        a.metric("Condition", condition)
        b.metric("Confidence", f"{binary[condition]:.1%}")
        c.metric("Fault type", fault)
        d.metric("Health index", f"{health:.0f}/100")
        st.caption("Health Index = 100 × model probability of Healthy. It is a model score, not remaining useful life.")
        st.subheader("Class probabilities")
        st.bar_chart(pd.Series(multiclass, name="probability"))

        first = windows[0]
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
        axes[0].plot(np.arange(len(first)) / 12000, first, linewidth=.7)
        axes[0].set(title="First 2048-sample window", xlabel="Seconds", ylabel="Amplitude")
        spectrum = np.abs(np.fft.rfft(first * np.hanning(len(first))))
        axes[1].plot(np.fft.rfftfreq(len(first), 1 / 12000), spectrum, linewidth=.7)
        axes[1].set(title="FFT", xlabel="Hz", ylabel="Amplitude")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(f"Prediction averages {len(windows)} non-overlapping window(s).")
except Exception as error:
    st.error(str(error))
