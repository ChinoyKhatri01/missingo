
import numpy as np
import pytest

from missingo import Signal
from missingo.signal_statistics.stats import standard_deviation


def test_standard_deviation_positive():
    signal = Signal(np.array([2, 4, 6, 8]), 1000)
    assert standard_deviation(signal) == pytest.approx(np.sqrt(5))


def test_standard_deviation_negative():
    signal = Signal(np.array([-2, -4, -6, -8]), 1000)
    assert standard_deviation(signal) == pytest.approx(np.sqrt(5))


def test_standard_deviation_mixed():
    signal = Signal(np.array([-2, 0, 2]), 1000)
    assert standard_deviation(signal) == pytest.approx(np.sqrt(8 / 3))


def test_standard_deviation_constant_signal():
    signal = Signal(np.array([5, 5, 5, 5]), 1000)
    assert standard_deviation(signal) == 0.0


def test_standard_deviation_zero():
    signal = Signal(np.array([0, 0, 0]), 1000)
    assert standard_deviation(signal) == 0.0


def test_standard_deviation_single_sample():
    signal = Signal(np.array([7]), 1000)
    assert standard_deviation(signal) == 0.0


def test_standard_deviation_complex():
    signal = Signal(np.array([1 + 1j, 3 + 1j]), 1000)
    assert standard_deviation(signal) == pytest.approx(1.0)


def test_standard_deviation_complex_multiple_samples():
    signal = Signal(np.array([1 + 2j, 3 + 4j, 5 + 6j]), 1000)
    assert standard_deviation(signal) == pytest.approx(np.sqrt(16 / 3))


def test_standard_deviation_invalid_input():
    with pytest.raises(TypeError):
        standard_deviation([1, 2, 3])


def test_standard_deviation_returns_native_float():
    signal = Signal(np.array([2.0, 4.0, 6.0]), 1000)
    result = standard_deviation(signal)

    assert isinstance(result, float)


def test_standard_deviation_does_not_modify_input():
    data = np.array([2, 4, 6, 8])
    signal = Signal(data, 1000)

    original = signal.data.copy()
    standard_deviation(signal)

    np.testing.assert_array_equal(signal.data, original)
