"""Shared preprocessing and feature configuration for training and inference."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIG_SCHEMA_VERSION = 1
FEATURE_VERSION = "cwru-signal-features-25-v1"
WINDOW_SIZE = 2048
TARGET_SAMPLING_RATE = 12000
SUPPORTED_INPUT_SAMPLING_RATES = (12000, 48000)
RESAMPLING_METHOD = "scipy.signal.decimate(factor=4, ftype='fir', zero_phase=True)"

FEATURE_NAMES = [
    "RMS", "Std", "Peak", "Kurtosis", "Crest_Factor",
    "FFT_Peak_Frequency", "Peak_to_Peak", "Mean", "Skewness",
    "Shape_Factor", "Impulse_Factor", "Margin_Factor",
    "FFT_Peak_Amplitude", "Spectral_Centroid", "Spectral_Spread",
    "Spectral_Entropy", "Spectral_Kurtosis",
    "Band_Energy_1", "Band_Energy_2", "Band_Energy_3", "Band_Energy_4",
    "Band_Energy_5", "Band_Energy_6", "Band_Energy_7", "Band_Energy_8",
]


def preprocessing_config():
    """Return a JSON-serializable copy of the inference contract."""
    return {
        "target_sampling_rate": TARGET_SAMPLING_RATE,
        "supported_input_sampling_rates": list(SUPPORTED_INPUT_SAMPLING_RATES),
        "window_size": WINDOW_SIZE,
        "window_overlap": 0,
        "incomplete_tail": "discard",
        "resampling_method": RESAMPLING_METHOD,
    }
