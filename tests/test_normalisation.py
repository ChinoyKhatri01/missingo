import pytest
import numpy as np

from missingo import Signal
from missingo.time_domain_operations.normalisation import normalisation


# --------------------------------------------------
# Basic normalization tests
# --------------------------------------------------

def test_normalisation_default_amplitude():
    signal = Signal([-2, 4, -6, 3], 1000)

    result = normalisation(signal)

    expected = np.array([-1/3, 2/3, -1, 1/2])

    np.testing.assert_allclose(result.data, expected)


def test_normalisation_custom_amplitude():
    signal = Signal([-2, 4, -6, 3], 1000)

    result = normalisation(signal, amplitude=5)

    expected = np.array([-5/3, 10/3, -5, 5/2])

    np.testing.assert_allclose(result.data, expected)


def test_normalisation_positive_signal():
    signal = Signal([2, 4, 6, 3], 1000)

    result = normalisation(signal)

    expected = np.array([1/3, 2/3, 1, 1/2])

    np.testing.assert_allclose(result.data, expected)


def test_normalisation_negative_amplitude():
    signal = Signal([-2, 4, -6, 3], 1000)

    with pytest.warns(UserWarning):
        result = normalisation(signal, amplitude=-2)

    expected = np.array([-2/3, 4/3, -2, 1])

    np.testing.assert_allclose(result.data, expected)


def test_normalisation_zero_amplitude():
    signal = Signal([-2, 4, -6, 3], 1000)

    result = normalisation(signal, amplitude=0)

    expected = np.zeros(4)

    np.testing.assert_allclose(result.data, expected)


# --------------------------------------------------
# Complex signal tests
# --------------------------------------------------

def test_normalisation_complex_signal():
    signal = Signal([3+4j, 6+8j, 1+2j], 1000)

    result = normalisation(signal)

    expected = np.array([
        (3+4j)/10,
        (6+8j)/10,
        (1+2j)/10
    ])

    np.testing.assert_allclose(result.data, expected)


def test_normalisation_complex_signal_custom_amplitude():
    signal = Signal([3+4j, 6+8j], 1000)

    result = normalisation(signal, amplitude=5)

    expected = np.array([
        1.5+2j,
        3+4j
    ])

    np.testing.assert_allclose(result.data, expected)


# --------------------------------------------------
# Input validation tests
# --------------------------------------------------

def test_normalisation_invalid_signal():
    with pytest.raises(TypeError):
        normalisation([1, 2, 3])


def test_normalisation_boolean_amplitude():
    signal = Signal([1, 2, 3], 1000)

    with pytest.raises(TypeError):
        normalisation(signal, amplitude=True)


@pytest.mark.parametrize("amplitude", [
    "5",
    None,
    [1, 2],
    2+3j,
])

def test_normalisation_invalid_amplitude_types(amplitude):
    signal = Signal([1, 2, 3], 1000)

    with pytest.raises(TypeError):
        normalisation(signal, amplitude=amplitude)


@pytest.mark.parametrize("amplitude", [
    np.nan,
    np.inf,
    -np.inf,
])

def test_normalisation_non_finite_amplitude(amplitude):
    signal = Signal([1, 2, 3], 1000)

    with pytest.raises(ValueError):
        normalisation(signal, amplitude=amplitude)


# --------------------------------------------------
# Zero-valued signal tests
# --------------------------------------------------

def test_normalisation_zero_valued_signal():
    signal = Signal([0, 0, 0, 0], 1000)

    with pytest.raises(ValueError):
        normalisation(signal)


# --------------------------------------------------
# Output properties
# --------------------------------------------------

def test_normalisation_returns_signal():
    signal = Signal([1, 2, 3], 1000)

    result = normalisation(signal)

    assert isinstance(result, Signal)


def test_normalisation_preserves_fs():
    signal = Signal([1, 2, 3], 2000)

    result = normalisation(signal)

    assert result.fs == signal.fs


def test_normalisation_preserves_sample_count():
    signal = Signal([1, 2, 3, 4], 1000)

    result = normalisation(signal)

    assert result.n_samples == signal.n_samples


def test_normalisation_does_not_modify_input():
    signal = Signal([2, 4, 6, 3], 1000)

    original = signal.data.copy()

    normalisation(signal)

    np.testing.assert_array_equal(signal.data, original)


def test_normalisation_returns_independent_signal():
    signal = Signal([1, 2, 3], 1000)

    result = normalisation(signal)

    assert result is not signal