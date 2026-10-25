import numpy as np
import pytest

from inference import prepare_windows_with_metadata


def test_12khz_passthrough_and_tail_metadata():
    windows, meta = prepare_windows_with_metadata(np.arange(4097.0), 12000)
    assert windows.shape == (2, 2048)
    assert meta["processed_sample_count"] == 4097
    assert meta["discarded_tail_samples"] == 1
    assert meta["processed_sampling_rate"] == 12000


def test_48khz_decimates_to_12khz():
    windows, meta = prepare_windows_with_metadata(np.arange(8192.0), 48000)
    assert windows.shape[1] == 2048
    assert meta["source_sampling_rate"] == 48000
    assert meta["processed_sample_count"] < 8192
    assert meta["processed_sampling_rate"] == 12000


@pytest.mark.parametrize("rate", [8000, 96000, 0])
def test_rejects_unsupported_sampling_rate(rate):
    with pytest.raises(ValueError, match="12,000 or 48,000"):
        prepare_windows_with_metadata(np.zeros(4096), rate)

