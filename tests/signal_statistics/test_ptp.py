
import numpy as np
import pytest

from missingo import Signal
from missingo.signal_statistics.stats import peak_to_peak


def test_peak_to_peak_positive_values():
    signal = Signal(np.array([2, 8, 4, 10, 6]), 1000)

    assert peak_to_peak(signal) == 8


def test_peak_to_peak_negative_values():
    signal = Signal(np.array([-8, -2, -10, -4]), 1000)

    assert peak_to_peak(signal) == 8


def test_peak_to_peak_mixed_values():
    signal = Signal(np.array([-5, 0, 7, -2, 3]), 1000)

    assert peak_to_peak(signal) == 12


def test_peak_to_peak_single_sample():
    signal = Signal(np.array([7]), 1000)

    assert peak_to_peak(signal) == 0


def test_peak_to_peak_zero_signal():
    signal = Signal(np.zeros(5), 1000)

    assert peak_to_peak(signal) == 0


def test_peak_to_peak_complex_signal():
    signal = Signal(np.array([1+2j, 3+4j]), 1000)

    with pytest.raises(TypeError):
        peak_to_peak(signal)


def test_peak_to_peak_invalid_input():
    with pytest.raises(TypeError):
        peak_to_peak([1, 2, 3])


def test_peak_to_peak_returns_native_python_scalar():
    signal = Signal(np.array([2, 8, 4]), 1000)

    result = peak_to_peak(signal)

    assert type(result) is int


def test_peak_to_peak_returns_native_float():
    signal = Signal(np.array([2.5, 8.5, 4.5]), 1000)

    result = peak_to_peak(signal)

    assert type(result) is float


def test_peak_to_peak_does_not_modify_original_signal():
    signal = Signal(np.array([2, 8, 4, 10, 6]), 1000)
    original_copy = signal.data.copy()

    peak_to_peak(signal)

    np.testing.assert_array_equal(signal.data, original_copy)
