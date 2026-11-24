
import numpy as np
import pytest

from missingo import Signal
from missingo.signal_statistics.stats import minimum


def test_minimum_positive_values():
    signal = Signal(np.array([2, 8, 4, 10, 6]), 1000)

    assert minimum(signal) == 2


def test_minimum_negative_values():
    signal = Signal(np.array([-8, -2, -10, -4]), 1000)

    assert minimum(signal) == -10


def test_minimum_mixed_values():
    signal = Signal(np.array([-5, 0, 7, -2, 3]), 1000)

    assert minimum(signal) == -5


def test_minimum_single_sample():
    signal = Signal(np.array([7]), 1000)

    assert minimum(signal) == 7


def test_minimum_zero_signal():
    signal = Signal(np.zeros(5), 1000)

    assert minimum(signal) == 0


def test_minimum_complex_signal():
    signal = Signal(np.array([1+2j, 3+4j]), 1000)

    with pytest.raises(TypeError):
        minimum(signal)


def test_minimum_invalid_input():
    with pytest.raises(TypeError):
        minimum([1, 2, 3])


def test_minimum_returns_native_python_scalar():
    signal = Signal(np.array([2, 8, 4]), 1000)

    result = minimum(signal)

    assert type(result) is int


def test_minimum_returns_native_float():
    signal = Signal(np.array([2.5, 8.5, 4.5]), 1000)

    result = minimum(signal)

    assert type(result) is float


def test_minimum_does_not_modify_original_signal():
    signal = Signal(np.array([2, 8, 4, 10, 6]), 1000)
    original_copy = signal.data.copy()

    minimum(signal)

    np.testing.assert_array_equal(signal.data, original_copy)
