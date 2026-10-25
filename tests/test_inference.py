import numpy as np

from inference import analyze_signal


def test_inference_contract_has_recording_and_window_outputs():
    result = analyze_signal(np.sin(np.linspace(0, 100, 4096)), 12000, "binary", "rf", "test.csv")
    assert result.task == "binary"
    assert result.model_family == "rf"
    assert result.valid_window_count == 2
    assert result.input_filename == "test.csv"
    assert set(result.probabilities) == set(result.classes)
    assert len(result.window_predictions) == 2
    assert len(result.window_probabilities) == 2
    assert len(result.first_window_features) == 25
    assert 0 <= result.confidence <= 1


def test_48khz_inference_records_resampling_warning():
    result = analyze_signal(np.sin(np.linspace(0, 100, 8192)), 48000, "multiclass", "svm")
    assert result.processed_sampling_rate == 12000
    assert any("resampled" in warning for warning in result.warnings)

