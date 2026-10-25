import numpy as np
import pytest

from inference import prepare_windows


def test_exact_and_multiple_window_counts():
    assert prepare_windows(np.zeros(2048), 12000).shape == (1, 2048)
    assert prepare_windows(np.zeros(4096), 12000).shape == (2, 2048)


def test_short_signal_is_rejected():
    with pytest.raises(ValueError, match="at least 2048"):
        prepare_windows(np.zeros(2047), 12000)

