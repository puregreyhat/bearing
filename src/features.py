"""Feature extraction for a single vibration window."""

import numpy as np
from scipy.stats import kurtosis, skew


def extract_features(x, sampling_rate):
    x = np.asarray(x, dtype=float)
    absolute = np.abs(x)
    rms = np.sqrt(np.mean(x * x))
    mean_abs = np.mean(absolute)
    sqrt_abs_mean = np.mean(np.sqrt(absolute))
    peak = np.max(absolute)

    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) * 2 / np.sum(np.hanning(len(x)))
    frequencies = np.fft.rfftfreq(len(x), 1 / sampling_rate)
    spectrum[0] = 0
    power = spectrum * spectrum
    total_power = power.sum()
    probabilities = power / total_power if total_power else np.zeros_like(power)
    centroid = np.sum(frequencies * probabilities)
    spread = np.sqrt(np.sum((frequencies - centroid) ** 2 * probabilities))
    nonzero = probabilities[probabilities > 0]
    bands = np.array_split(power, 8)

    values = {
        "RMS": rms, "Std": np.std(x), "Peak": peak,
        "Peak_to_Peak": np.ptp(x), "Mean": np.mean(x),
        "Kurtosis": kurtosis(x, fisher=False), "Skewness": skew(x),
        "Crest_Factor": peak / rms if rms else 0,
        "Shape_Factor": rms / mean_abs if mean_abs else 0,
        "Impulse_Factor": peak / mean_abs if mean_abs else 0,
        "Margin_Factor": peak / (sqrt_abs_mean ** 2) if sqrt_abs_mean else 0,
        "FFT_Peak_Frequency": frequencies[np.argmax(spectrum)],
        "FFT_Peak_Amplitude": np.max(spectrum),
        "Spectral_Centroid": centroid, "Spectral_Spread": spread,
        "Spectral_Entropy": -np.sum(nonzero * np.log2(nonzero)) if nonzero.size else 0,
        "Spectral_Kurtosis": kurtosis(power, fisher=False) if total_power else 0,
    }
    values.update({f"Band_Energy_{i + 1}": band.sum() for i, band in enumerate(bands)})
    if not np.isfinite(list(values.values())).all():
        raise ValueError("feature extraction produced NaN or infinity")
    return values
