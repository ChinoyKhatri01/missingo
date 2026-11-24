
import numpy as np
import pytest

from missingo import Signal
from missingo.signal_statistics.stats import mean


def test_mean_positive_values():
    signal = Signal(np.array([2, 4, 6, 8, 10]), 1000)

    assert mean(signal) == 6.0


def test_mean_positive_and_negative_values():
    signal = Signal(np.array([-5, -2, 3, 4]), 1000)

    assert mean(signal) == 0.0


def test_mean_zero_signal():
    signal = Signal(np.zeros(5), 1000)

    assert mean(signal) == 0.0


def test_mean_complex_signal():
    signal = Signal(np.array([1+2j, 3+4j]), 1000)

    assert mean(signal) == 2+3j


def test_mean_single_sample():
    signal = Signal(np.array([7]), 1000)

    assert mean(signal) == 7.0


def test_mean_invalid_input():
    with pytest.raises(TypeError):
        mean([1, 2, 3])


def test_mean_returns_native_python_scalar():
    signal = Signal(np.array([2, 4, 6]), 1000)

    result = mean(signal)

    assert type(result) is float


def test_mean_does_not_modify_original_signal():
    original_data = np.array([2, 4, 6, 8])
    signal = Signal(original_data, 1000)

    original_copy = signal.data.copy()

    mean(signal)

    np.testing.assert_array_equal(signal.data, original_copy)
