
import numpy as np
import pytest

from missingo import Signal
from missingo.signal_statistics.stats import rms


def test_rms_positive():
    signal = Signal(np.array([3, 4]), 1000)
    assert rms(signal) == pytest.approx(np.sqrt(12.5))


def test_rms_negative():
    signal = Signal(np.array([-3, -4]), 1000)
    assert rms(signal) == pytest.approx(np.sqrt(12.5))


def test_rms_mixed():
    signal = Signal(np.array([-3, 4, -5, 2]), 1000)
    assert rms(signal) == pytest.approx(np.sqrt(13.5))


def test_rms_zero():
    signal = Signal(np.array([0, 0, 0]), 1000)
    assert rms(signal) == 0.0


def test_rms_single_sample():
    signal = Signal(np.array([-7]), 1000)
    assert rms(signal) == 7.0


def test_rms_complex():
    signal = Signal(np.array([3 + 4j, 0 + 0j]), 1000)
    assert rms(signal) == pytest.approx(np.sqrt(12.5))


def test_rms_complex_multiple_samples():
    signal = Signal(np.array([3 + 4j, 5 + 12j]), 1000)
    assert rms(signal) == pytest.approx(np.sqrt(97))


def test_rms_invalid_input():
    with pytest.raises(TypeError):
        rms([1, 2, 3])


def test_rms_returns_native_float():
    signal = Signal(np.array([3.0, 4.0]), 1000)
    result = rms(signal)

    assert isinstance(result, float)


def test_rms_does_not_modify_input():
    data = np.array([-3, 4, -5, 2])
    signal = Signal(data, 1000)

    original = signal.data.copy()
    rms(signal)

    np.testing.assert_array_equal(signal.data, original)
