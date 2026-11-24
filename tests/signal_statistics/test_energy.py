
import numpy as np
import pytest

from missingo import Signal
from missingo.signal_statistics.stats import energy


def test_energy_positive():
    signal = Signal(np.array([3, 4]), 1000)
    assert energy(signal) == 25


def test_energy_negative():
    signal = Signal(np.array([-3, -4]), 1000)
    assert energy(signal) == 25


def test_energy_mixed():
    signal = Signal(np.array([-3, 4, -5, 2]), 1000)
    assert energy(signal) == 54


def test_energy_zero():
    signal = Signal(np.array([0, 0, 0]), 1000)
    assert energy(signal) == 0


def test_energy_single_sample():
    signal = Signal(np.array([-7]), 1000)
    assert energy(signal) == 49


def test_energy_complex():
    signal = Signal(np.array([3 + 4j, 0 + 0j]), 1000)
    assert energy(signal) == 25


def test_energy_complex_multiple_samples():
    signal = Signal(np.array([3 + 4j, 5 + 12j]), 1000)
    assert energy(signal) == 194


def test_energy_invalid_input():
    with pytest.raises(TypeError):
        energy([1, 2, 3])


def test_energy_returns_native_float():
    signal = Signal(np.array([3.0, 4.0]), 1000)
    result = energy(signal)

    assert isinstance(result, float)


def test_energy_returns_native_int():
    signal = Signal(np.array([3, 4]), 1000)
    result = energy(signal)

    assert isinstance(result, int)


def test_energy_does_not_modify_input():
    data = np.array([-3, 4, -5, 2])
    signal = Signal(data, 1000)

    original = signal.data.copy()
    energy(signal)

    np.testing.assert_array_equal(signal.data, original)

