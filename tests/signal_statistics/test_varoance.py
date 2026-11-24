
import numpy as np
import pytest

from missingo import Signal
from missingo.signal_statistics.stats import variance


def test_variance_positive():
    signal = Signal(np.array([2, 4, 6, 8]), 1000)
    assert variance(signal) == pytest.approx(5.0)


def test_variance_negative():
    signal = Signal(np.array([-2, -4, -6, -8]), 1000)
    assert variance(signal) == pytest.approx(5.0)


def test_variance_mixed():
    signal = Signal(np.array([-2, 0, 2]), 1000)
    assert variance(signal) == pytest.approx(8 / 3)


def test_variance_constant_signal():
    signal = Signal(np.array([5, 5, 5, 5]), 1000)
    assert variance(signal) == 0.0


def test_variance_zero():
    signal = Signal(np.array([0, 0, 0]), 1000)
    assert variance(signal) == 0.0


def test_variance_single_sample():
    signal = Signal(np.array([7]), 1000)
    assert variance(signal) == 0.0


def test_variance_complex():
    signal = Signal(np.array([1 + 1j, 3 + 1j]), 1000)
    assert variance(signal) == pytest.approx(1.0)


def test_variance_complex_multiple_samples():
    signal = Signal(np.array([1 + 2j, 3 + 4j, 5 + 6j]), 1000)
    assert variance(signal) == pytest.approx(16 / 3)


def test_variance_invalid_input():
    with pytest.raises(TypeError):
        variance([1, 2, 3])


def test_variance_returns_native_float():
    signal = Signal(np.array([2.0, 4.0, 6.0]), 1000)
    result = variance(signal)

    assert isinstance(result, float)


def test_variance_does_not_modify_input():
    data = np.array([2, 4, 6, 8])
    signal = Signal(data, 1000)

    original = signal.data.copy()
    variance(signal)

    np.testing.assert_array_equal(signal.data, original)
