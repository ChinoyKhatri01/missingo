
import numpy as np
import pytest

from missingo import Signal
from missingo.signal_statistics.stats import median


def test_median_odd_number_of_samples():
    signal = Signal(np.array([8, 2, 10, 4, 6]), 1000)

    assert median(signal) == 6.0


def test_median_even_number_of_samples():
    signal = Signal(np.array([2, 4, 6, 8]), 1000)

    assert median(signal) == 5.0


def test_median_negative_values():
    signal = Signal(np.array([-8, -2, -10, -4, -6]), 1000)

    assert median(signal) == -6.0


def test_median_unsorted_values():
    signal = Signal(np.array([10, 1, 7, 3, 5]), 1000)

    assert median(signal) == 5.0


def test_median_single_sample():
    signal = Signal(np.array([7]), 1000)

    assert median(signal) == 7.0


def test_median_zero_signal():
    signal = Signal(np.zeros(5), 1000)

    assert median(signal) == 0.0


def test_median_complex_signal():
    signal = Signal(np.array([1+2j, 3+4j]), 1000)

    with pytest.raises(TypeError):
        median(signal)


def test_median_invalid_input():
    with pytest.raises(TypeError):
        median([1, 2, 3])


def test_median_returns_native_python_scalar():
    signal = Signal(np.array([2, 4, 6]), 1000)

    result = median(signal)

    assert type(result) is float


def test_median_does_not_modify_original_signal():
    signal = Signal(np.array([8, 2, 10, 4, 6]), 1000)
    original_copy = signal.data.copy()

    median(signal)

    np.testing.assert_array_equal(signal.data, original_copy)
