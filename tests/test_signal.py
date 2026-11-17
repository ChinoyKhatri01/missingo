import numpy as np
from missingo import Signal
import pytest
import re

# 1. Valid signal construction

def test_signal_creation_with_integer_list():
    signal = Signal([10, 20, 30, 40, 50], 1000)
    assert signal.n_samples == 5
    assert signal.fs == 1000

def test_signal_creation_with_float_list():
    signal = Signal([1.5, 2.5, 3.5], 2000)
    assert signal.n_samples == 3
    assert signal.fs == 2000

def test_signal_creation_with_numpy_array():
    data = np.array([10, 20, 30, 40])
    signal = Signal(data, 1000)
    assert signal.n_samples == 4
    np.testing.assert_array_equal(signal.data, data)

def test_signal_creation_with_complex_data():
    data = [1 + 2j, 3 + 4j, 5 + 6j]
    signal = Signal(data, 1000)

    assert signal.n_samples == 3
    assert np.iscomplexobj(signal.data)

# 2. Invalid signal data

def test_multidimensional_signal():
    data = [[1, 2, 3], [4, 5, 6]]

    with pytest.raises(ValueError):
        Signal(data, 1000)

def test_boolean_signal():
    data = [True, False, True]

    with pytest.raises(ValueError):
        Signal(data, 1000)

def test_string_signal():
    data = ["one", "two", "three"]

    with pytest.raises(ValueError):
        Signal(data, 1000)

def test_signal_with_nan():
    data = [1, 2, np.nan, 4]
    with pytest.raises(ValueError):
        Signal(data, 1000)

def test_signal_with_positive_infinity():
    data = [1, 2, np.inf, 4]
    with pytest.raises(ValueError):
        Signal(data, 1000)

def test_signal_with_negative_infinity():
    data = [1, 2, -np.inf, 4]

    with pytest.raises(ValueError):
        Signal(data, 1000)

# 3. Sampling frequency validation

def test_signal_with_float_fs():
    signal = Signal([1, 2, 3], 1000.5)
    assert signal.fs == 1000.5

def test_signal_with_zero_fs():
    with pytest.raises(ValueError):
        Signal([1, 2, 3], 0)

def test_signal_with_negative_fs():
    with pytest.raises(ValueError):
        Signal([1, 2, 3], -1000)

def test_signal_with_boolean_fs():
    with pytest.raises(ValueError):
        Signal([1, 2, 3], True)

def test_signal_with_string_fs():
    with pytest.raises(ValueError):
        Signal([1, 2, 3], "1000")

def test_signal_with_infinite_fs():
    with pytest.raises(ValueError):
        Signal([1, 2, 3], np.inf)

def test_signal_with_nan_fs():
    with pytest.raises(ValueError):
        Signal([1, 2, 3], np.nan)

# 4. Signal properties

def test_n_samples():
    signal = Signal([10, 20, 30, 40, 50], 1000)
    assert signal.n_samples == 5

def test_fs():
    signal = Signal([10, 20, 30], 2000)
    assert signal.fs == 2000

def test_sampling_interval():
    signal = Signal([10, 20, 30], 1000)
    assert signal.sampling_interval == 0.001

def test_duration():
    signal = Signal([10, 20, 30, 40, 50], 1000)
    assert signal.duration == 0.005

def test_time_array():
    signal = Signal([10, 20, 30, 40, 50], 1000)
    expected_time = np.array([0, 0.001, 0.002, 0.003, 0.004])
    np.testing.assert_allclose(signal.time, expected_time)

def test_signal_length():
    signal = Signal([10, 20, 30, 40, 50], 1000)
    assert len(signal) == 5

def test_signal_data_is_independent():
    signal = Signal([10, 20, 30], 1000)

    returned_data = signal.data
    returned_data[0] = 999

    assert signal.data[0] == 10

def test_internal_data_is_readonly():
    signal = Signal([10, 20, 30], 1000)

    with pytest.raises(ValueError):
        signal._data[0] = 999

def test_time_array_is_readonly():
    signal = Signal([10, 20, 30], 1000)

    with pytest.raises(ValueError):
        signal.time[0] = 999

def test_mixed_boolean_and_integer_data():
    data = [10, 20, True, 40]

    with pytest.raises(ValueError):
        Signal(data, 1000)

def test_numpy_boolean_data():
    data = [10, 20, np.bool_(True), 40]

    with pytest.raises(ValueError):
        Signal(data, 1000)

def test_numpy_integer_fs():
    signal = Signal([10, 20, 30], np.int64(1000))
    assert signal.fs == 1000

def test_numpy_float_fs():
    signal = Signal([10, 20, 30], np.float32(1000.5))
    assert signal.fs == np.float32(1000.5)

def test_complex_fs():
    with pytest.raises(ValueError):
        Signal([10, 20, 30], 1000 + 2j)

def test_scalar_data():
    with pytest.raises(ValueError, match="data must be a list or a numpy array"):
        Signal(10, 1000)

# 5. Input type validation

def test_tuple_data():
    with pytest.raises(ValueError, match="data must be a list or a numpy array"):
        Signal((1, 2, 3), 1000)

def test_range_data():
    with pytest.raises(ValueError, match="data must be a list or a numpy array"):
        Signal(range(5), 1000)

def test_string_data_input():
    with pytest.raises(ValueError, match="data must be a list or a numpy array"):
        Signal("123", 1000)

# Additional input validation tests

def test_empty_numpy_array():
    with pytest.raises(ValueError, match="data must not be empty"):
        Signal(np.array([]), 1000)

def test_multidimensional_numpy_array():
    data = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="data must be a 1D array"):
        Signal(data, 1000)

def test_numpy_boolean_array():
    data = np.array([True, False, True])
    with pytest.raises(
        ValueError,
        match=re.escape("data must be numeric (float or int or complex)")
    ):
        Signal(data, 1000)

def test_mixed_integer_float_data():
    signal = Signal([1, 2.5, 3, 4.5], 1000)
    assert signal.n_samples == 4
    assert np.issubdtype(signal.data.dtype, np.floating)

def test_mixed_integer_complex_data():
    signal = Signal([1, 2 + 3j, 4], 1000)
    assert signal.n_samples == 3
    assert np.iscomplexobj(signal.data)

def test_numpy_boolean_fs():
    with pytest.raises(ValueError, match="fs must be a number"):
        Signal([1, 2, 3], np.bool_(True))

def test_numpy_negative_fs():
    with pytest.raises(ValueError):
        Signal([1, 2, 3], np.int64(-1000))

def test_original_list_modification_does_not_affect_signal():
    data = [10, 20, 30]
    signal = Signal(data, 1000)
    data[0] = 999
    np.testing.assert_array_equal(signal.data, [10, 20, 30])

def test_original_numpy_array_modification_does_not_affect_signal():
    data = np.array([10, 20, 30])
    signal = Signal(data, 1000)
    data[0] = 999
    np.testing.assert_array_equal(signal.data, [10, 20, 30])

def test_time_array_length_matches_n_samples():
    signal = Signal([1, 2, 3, 4, 5, 6], 2000)
    assert len(signal.time) == signal.n_samples

def test_time_array_with_different_sampling_frequency():
    signal = Signal([1, 2, 3, 4], 2000)
    expected = np.array([0, 0.0005, 0.001, 0.0015])
    np.testing.assert_allclose(signal.time, expected)

def test_signal_repr():
    signal = Signal([1, 2, 3], 1000)

    result = repr(signal)

    assert "Signal" in result
    assert "n_samples" in result
    assert "fs" in result
    assert "duration" in result
    assert "dtype" in result
    assert "ndim" in result

def test_modifying_returned_data_does_not_affect_signal():
    signal = Signal([10, 20, 30], 1000)
    data = signal.data
    data[0] = 999
    np.testing.assert_array_equal(signal.data, [10, 20, 30])

def test_returned_data_is_writable():
    signal = Signal([10, 20, 30], 1000)
    data = signal.data
    assert data.flags.writeable

def test_time_array_is_read_only():
    signal = Signal([10, 20, 30], 1000)
    with pytest.raises(ValueError):
        signal.time[0] = 999

def test_fs_is_stored_as_float():
    signal = Signal([1, 2, 3], 1000)
    assert isinstance(signal.fs, float)
    assert signal.fs == 1000.0

