import numpy as np

from config import FEATURE_NAMES
from features import extract_features


def test_feature_contract_is_finite_and_complete():
    features = extract_features(np.sin(np.linspace(0, 20, 2048)), 12000)
    assert list(features) == [
        "RMS", "Std", "Peak", "Peak_to_Peak", "Mean", "Kurtosis", "Skewness",
        "Crest_Factor", "Shape_Factor", "Impulse_Factor", "Margin_Factor",
        "FFT_Peak_Frequency", "FFT_Peak_Amplitude", "Spectral_Centroid",
        "Spectral_Spread", "Spectral_Entropy", "Spectral_Kurtosis",
        "Band_Energy_1", "Band_Energy_2", "Band_Energy_3", "Band_Energy_4",
        "Band_Energy_5", "Band_Energy_6", "Band_Energy_7", "Band_Energy_8",
    ]
    assert set(features) == set(FEATURE_NAMES)
    assert np.isfinite(list(features.values())).all()


def test_constant_signal_has_stable_zero_safe_features():
    features = extract_features(np.ones(2048), 12000)
    assert features["RMS"] == 1
    assert features["Crest_Factor"] == 1
    assert np.isfinite(list(features.values())).all()

