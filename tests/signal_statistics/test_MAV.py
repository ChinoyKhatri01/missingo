
import numpy as np
import pytest

from missingo import Signal
from missingo.signal_statistics.stats import mean_absolute_value


def test_mean_absolute_value_positive():
    signal = Signal(np.array([2, 4, 6, 8]), 1000)
    assert mean_absolute_value(signal) == 5.0


def test_mean_absolute_value_negative():
    signal = Signal(np.array([-2, -4, -6, -8]), 1000)
    assert mean_absolute_value(signal) == 5.0


def test_mean_absolute_value_mixed():
    signal = Signal(np.array([-4, 2, -6, 8]), 1000)
    assert mean_absolute_value(signal) == 5.0


def test_mean_absolute_value_cancellation():
    signal = Signal(np.array([-5, 5]), 1000)
    assert mean_absolute_value(signal) == 5.0


def test_mean_absolute_value_zero():
    signal = Signal(np.array([0, 0, 0]), 1000)
    assert mean_absolute_value(signal) == 0.0


def test_mean_absolute_value_single_sample():
    signal = Signal(np.array([-7]), 1000)
    assert mean_absolute_value(signal) == 7.0


def test_mean_absolute_value_complex():
    signal = Signal(np.array([3 + 4j, 0 + 0j]), 1000)
    assert mean_absolute_value(signal) == 2.5


def test_mean_absolute_value_complex_magnitudes():
    signal = Signal(np.array([3 + 4j, 5 + 12j]), 1000)
    assert mean_absolute_value(signal) == 9.0


def test_mean_absolute_value_invalid_input():
    with pytest.raises(TypeError):
        mean_absolute_value([1, 2, 3])


def test_mean_absolute_value_returns_native_float():
    signal = Signal(np.array([1.0, 2.0, 3.0]), 1000)
    result = mean_absolute_value(signal)

    assert isinstance(result, float)


def test_mean_absolute_value_does_not_modify_input():
    data = np.array([-2, 4, -6, 8])
    signal = Signal(data, 1000)

    original = signal.data.copy()
    mean_absolute_value(signal)

    np.testing.assert_array_equal(signal.data, original)
