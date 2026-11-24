
import numpy as np
import pytest

from missingo import Signal
from missingo.signal_statistics.stats import average_power


def test_average_power_positive():
    signal = Signal(np.array([3, 4]), 1000)
    assert average_power(signal) == 12.5


def test_average_power_negative():
    signal = Signal(np.array([-3, -4]), 1000)
    assert average_power(signal) == 12.5


def test_average_power_mixed():
    signal = Signal(np.array([-3, 4, -5, 2]), 1000)
    assert average_power(signal) == 13.5


def test_average_power_zero():
    signal = Signal(np.array([0, 0, 0]), 1000)
    assert average_power(signal) == 0.0


def test_average_power_single_sample():
    signal = Signal(np.array([-7]), 1000)
    assert average_power(signal) == 49.0


def test_average_power_complex():
    signal = Signal(np.array([3 + 4j, 0 + 0j]), 1000)
    assert average_power(signal) == 12.5


def test_average_power_complex_multiple_samples():
    signal = Signal(np.array([3 + 4j, 5 + 12j]), 1000)
    assert average_power(signal) == 97.0


def test_average_power_invalid_input():
    with pytest.raises(TypeError):
        average_power([1, 2, 3])


def test_average_power_returns_native_float():
    signal = Signal(np.array([3, 4]), 1000)
    result = average_power(signal)

    assert isinstance(result, float)


def test_average_power_does_not_modify_input():
    data = np.array([-3, 4, -5, 2])
    signal = Signal(data, 1000)

    original = signal.data.copy()
    average_power(signal)

    np.testing.assert_array_equal(signal.data, original)


def test_average_power_equals_rms_squared():
    signal = Signal(np.array([3, 4, -5, 2]), 1000)

    from missingo.signal_statistics.stats import rms

    assert average_power(signal) == pytest.approx(rms(signal) ** 2)
