import io

import numpy as np
import pytest
from scipy.io import savemat

from inference import read_signal


def test_loads_csv_npy_and_mat(tmp_path):
    values = np.arange(16, dtype=float)
    csv_path = tmp_path / "signal.csv"
    npy_path = tmp_path / "signal.npy"
    mat_path = tmp_path / "097.mat"
    np.savetxt(csv_path, values, delimiter=",")
    np.save(npy_path, values)
    savemat(mat_path, {"X097_DE_time": values.reshape(-1, 1)})
    np.testing.assert_array_equal(read_signal(csv_path, csv_path.name), values)
    np.testing.assert_array_equal(read_signal(npy_path, npy_path.name), values)
    np.testing.assert_array_equal(read_signal(mat_path, mat_path.name), values)


@pytest.mark.parametrize("name, payload, message", [
    ("empty.csv", "", "empty"),
    ("nan.csv", "1,nan,3", "NaN or infinity"),
    ("inf.csv", "1,inf,3", "NaN or infinity"),
    ("bad.txt", "1,2,3", "csv, .mat, or .npy"),
])
def test_rejects_invalid_inputs(tmp_path, name, payload, message):
    path = tmp_path / name
    path.write_text(payload)
    with pytest.raises((ValueError, OSError), match=message):
        read_signal(path, name)


def test_rejects_ambiguous_mat(tmp_path):
    path = tmp_path / "ambiguous.mat"
    savemat(path, {"X001_DE_time": [1, 2], "X002_DE_time": [3, 4]})
    with pytest.raises(ValueError, match="ambiguous"):
        read_signal(path, path.name)

