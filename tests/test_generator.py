
import numpy as np
import pytest
import warnings
from missingo import Signal, generator


# --------------------------------------------------
# Basic functionality
# --------------------------------------------------

def test_sine_returns_signal():
    result = generator.sine(5, 100, 100)

    assert isinstance(result, Signal)


def test_sine_output_properties():
    result = generator.sine(5, 100, 100)

    assert result.fs == 100.0
    assert result.n_samples == 100
    assert result.duration == 1.0
    assert len(result.time) == 100


def test_sine_default_parameters():
    result = generator.sine(5, 100, 100)

    expected = np.sin(2 * np.pi * 5 * np.arange(100) / 100)

    np.testing.assert_allclose(result.data, expected)


def test_sine_exact_sample_values():
    result = generator.sine(5, 20, 4)

    expected = np.array([0, 1, 0, -1])

    np.testing.assert_allclose(result.data, expected, atol=1e-12)


def test_sine_custom_parameters():
    result = generator.sine(
        frequency=5,
        fs=100,
        n_samples=100,
        amplitude=2,
        phase=np.pi / 4,
        offset=3,
    )

    n = np.arange(100)
    expected = 2 * np.sin(2 * np.pi * 5 * n / 100 + np.pi / 4) + 3

    np.testing.assert_allclose(result.data, expected)


# --------------------------------------------------
# Frequency validation
# --------------------------------------------------

@pytest.mark.parametrize("frequency", [
    True,
    False,
    np.bool_(True),
    "5",
    1 + 2j,
])
def test_invalid_frequency_type(frequency):
    with pytest.raises(TypeError):
        generator.sine(frequency, 100, 10)


@pytest.mark.parametrize("frequency", [
    -1,
    -0.5,
])
def test_negative_frequency(frequency):
    with pytest.raises(ValueError):
        generator.sine(frequency, 100, 10)


@pytest.mark.parametrize("frequency", [
    np.nan,
    np.inf,
    -np.inf,
])
def test_non_finite_frequency(frequency):
    with pytest.raises(ValueError):
        generator.sine(frequency, 100, 10)


def test_zero_frequency_is_allowed():
    result = generator.sine(0, 100, 5, amplitude=2, phase=np.pi / 2)

    np.testing.assert_allclose(result.data, np.full(5, 2.0))


# --------------------------------------------------
# Nyquist frequency and aliasing warnings
# --------------------------------------------------

def test_frequency_above_nyquist_warns():
    with pytest.warns(UserWarning, match="Nyquist"):
        result = generator.sine(60, 100, 10)

    assert isinstance(result, Signal)


def test_frequency_at_nyquist_does_not_warn():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        generator.sine(50, 100, 10)

    assert len(recorded) == 0


def test_frequency_below_nyquist_does_not_warn():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        generator.sine(40, 100, 10)

    assert len(recorded) == 0


# --------------------------------------------------
# Sampling frequency validation
# Delegated to the Signal class
# --------------------------------------------------

@pytest.mark.parametrize("fs", [
    True,
    False,
    np.bool_(True),
    "100",
    1 + 2j,
])
def test_invalid_fs_type(fs):
    with pytest.raises(TypeError):
        generator.sine(5, fs, 10)


@pytest.mark.parametrize("fs", [
    0,
    -100,
    np.nan,
    np.inf,
    -np.inf,
])
def test_invalid_fs_value(fs):
    with pytest.raises(ValueError):
        generator.sine(5, fs, 10)


# --------------------------------------------------
# n_samples validation
# --------------------------------------------------

@pytest.mark.parametrize("n_samples", [
    True,
    False,
    np.bool_(True),
    "10",
    1 + 2j,
])
def test_invalid_n_samples_type(n_samples):
    with pytest.raises(TypeError):
        generator.sine(5, 100, n_samples)


@pytest.mark.parametrize("n_samples", [
    0,
    -1,
    -10,
])
def test_non_positive_n_samples(n_samples):
    with pytest.raises(ValueError):
        generator.sine(5, 100, n_samples)


@pytest.mark.parametrize("n_samples", [
    np.nan,
    np.inf,
    -np.inf,
])
def test_non_finite_n_samples(n_samples):
    with pytest.raises(ValueError):
        generator.sine(5, 100, n_samples)


def test_fractional_n_samples_are_rounded():
    with pytest.warns(UserWarning, match="Fractional"):
        result = generator.sine(5, 100, 5.7)

    assert result.n_samples == 6


def test_integer_float_n_samples_are_accepted_without_warning():
    result = generator.sine(5, 100, 5.0)

    assert result.n_samples == 5


def test_fractional_n_samples_rounding_to_zero_raises():
    with pytest.warns(UserWarning, match="Fractional"):
        with pytest.raises(ValueError):
            generator.sine(5, 100, 0.4)


# --------------------------------------------------
# Amplitude validation
# --------------------------------------------------

@pytest.mark.parametrize("amplitude", [
    True,
    False,
    np.bool_(True),
    "2",
    1 + 2j,
])
def test_invalid_amplitude_type(amplitude):
    with pytest.raises(TypeError):
        generator.sine(5, 100, 10, amplitude=amplitude)


@pytest.mark.parametrize("amplitude", [
    np.nan,
    np.inf,
    -np.inf,
])
def test_non_finite_amplitude(amplitude):
    with pytest.raises(ValueError):
        generator.sine(5, 100, 10, amplitude=amplitude)


def test_zero_amplitude_is_allowed():
    result = generator.sine(5, 100, 10, amplitude=0, offset=3)

    np.testing.assert_allclose(result.data, np.full(10, 3.0))


def test_negative_amplitude_is_allowed():
    result = generator.sine(5, 20, 4, amplitude=-1)

    expected = -np.sin(2 * np.pi * 5 * np.arange(4) / 20)

    np.testing.assert_allclose(result.data, expected)


# --------------------------------------------------
# Phase validation
# --------------------------------------------------

@pytest.mark.parametrize("phase", [
    True,
    False,
    np.bool_(True),
    "1.5",
    1 + 2j,
])
def test_invalid_phase_type(phase):
    with pytest.raises(TypeError):
        generator.sine(5, 100, 10, phase=phase)


@pytest.mark.parametrize("phase", [
    np.nan,
    np.inf,
    -np.inf,
])
def test_non_finite_phase(phase):
    with pytest.raises(ValueError):
        generator.sine(5, 100, 10, phase=phase)


def test_negative_phase_is_allowed():
    result = generator.sine(0, 100, 5, amplitude=2, phase=-np.pi / 2)

    np.testing.assert_allclose(result.data, np.full(5, -2.0))


def test_phase_greater_than_two_pi_is_allowed():
    result = generator.sine(
        0, 100, 5, amplitude=2, phase=5 * np.pi / 2
    )

    np.testing.assert_allclose(result.data, np.full(5, 2.0))


# --------------------------------------------------
# Offset validation
# --------------------------------------------------

@pytest.mark.parametrize("offset", [
    True,
    False,
    np.bool_(True),
    "3",
    1 + 2j,
])
def test_invalid_offset_type(offset):
    with pytest.raises(TypeError):
        generator.sine(5, 100, 10, offset=offset)


@pytest.mark.parametrize("offset", [
    np.nan,
    np.inf,
    -np.inf,
])
def test_non_finite_offset(offset):
    with pytest.raises(ValueError):
        generator.sine(5, 100, 10, offset=offset)


def test_negative_offset_is_allowed():
    result = generator.sine(0, 100, 5, offset=-3)

    np.testing.assert_allclose(result.data, np.full(5, -3.0))


# ==================================================
# COSINE GENERATOR TESTS
# ==================================================


# --------------------------------------------------
# Basic functionality
# --------------------------------------------------

def test_cosine_returns_signal():
    result = generator.cosine(5, 100, 100)

    assert isinstance(result, Signal)


def test_cosine_output_properties():
    result = generator.cosine(5, 100, 100)

    assert result.fs == 100.0
    assert result.n_samples == 100
    assert result.duration == 1.0
    assert len(result.time) == 100


def test_cosine_default_parameters():
    result = generator.cosine(5, 100, 100)

    expected = np.cos(2 * np.pi * 5 * np.arange(100) / 100)

    np.testing.assert_allclose(result.data, expected)


def test_cosine_exact_sample_values():
    result = generator.cosine(5, 20, 4)

    expected = np.array([1, 0, -1, 0])

    np.testing.assert_allclose(result.data, expected, atol=1e-12)


def test_cosine_custom_parameters():
    result = generator.cosine(
        frequency=5,
        fs=100,
        n_samples=100,
        amplitude=2,
        phase=np.pi / 4,
        offset=3,
    )

    n = np.arange(100)
    expected = 2 * np.cos(2 * np.pi * 5 * n / 100 + np.pi / 4) + 3

    np.testing.assert_allclose(result.data, expected)


def test_cosine_first_sample_equals_amplitude_plus_offset():
    result = generator.cosine(
        frequency=5,
        fs=100,
        n_samples=10,
        amplitude=4,
        offset=2,
    )

    assert result.data[0] == pytest.approx(6)


# --------------------------------------------------
# Frequency validation
# --------------------------------------------------

@pytest.mark.parametrize("frequency", [
    True,
    False,
    np.bool_(True),
    "5",
    1 + 2j,
])
def test_cosine_invalid_frequency_type(frequency):
    with pytest.raises(TypeError):
        generator.cosine(frequency, 100, 10)


@pytest.mark.parametrize("frequency", [
    -1,
    -0.5,
])
def test_cosine_negative_frequency(frequency):
    with pytest.raises(ValueError):
        generator.cosine(frequency, 100, 10)


@pytest.mark.parametrize("frequency", [
    np.nan,
    np.inf,
    -np.inf,
])
def test_cosine_non_finite_frequency(frequency):
    with pytest.raises(ValueError):
        generator.cosine(frequency, 100, 10)


def test_cosine_zero_frequency_is_allowed():
    result = generator.cosine(
        0, 100, 5,
        amplitude=2,
        offset=3,
    )

    np.testing.assert_allclose(result.data, np.full(5, 5.0))


# --------------------------------------------------
# Nyquist frequency and aliasing warnings
# --------------------------------------------------

def test_cosine_frequency_above_nyquist_warns():
    with pytest.warns(UserWarning, match="Nyquist"):
        result = generator.cosine(60, 100, 10)

    assert isinstance(result, Signal)


def test_cosine_frequency_at_nyquist_does_not_warn():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        generator.cosine(50, 100, 10)

    assert len(recorded) == 0


def test_cosine_frequency_below_nyquist_does_not_warn():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        generator.cosine(40, 100, 10)

    assert len(recorded) == 0


# --------------------------------------------------
# Sampling frequency validation
# --------------------------------------------------

@pytest.mark.parametrize("fs", [
    True,
    False,
    np.bool_(True),
    "100",
    1 + 2j,
])
def test_cosine_invalid_fs_type(fs):
    with pytest.raises(TypeError):
        generator.cosine(5, fs, 10)


@pytest.mark.parametrize("fs", [
    0,
    -100,
    np.nan,
    np.inf,
    -np.inf,
])
def test_cosine_invalid_fs_value(fs):
    with pytest.raises(ValueError):
        generator.cosine(5, fs, 10)


# --------------------------------------------------
# n_samples validation
# --------------------------------------------------

@pytest.mark.parametrize("n_samples", [
    True,
    False,
    np.bool_(True),
    "10",
    1 + 2j,
])
def test_cosine_invalid_n_samples_type(n_samples):
    with pytest.raises(TypeError):
        generator.cosine(5, 100, n_samples)


@pytest.mark.parametrize("n_samples", [
    0,
    -1,
    -10,
])
def test_cosine_non_positive_n_samples(n_samples):
    with pytest.raises(ValueError):
        generator.cosine(5, 100, n_samples)


@pytest.mark.parametrize("n_samples", [
    np.nan,
    np.inf,
    -np.inf,
])
def test_cosine_non_finite_n_samples(n_samples):
    with pytest.raises(ValueError):
        generator.cosine(5, 100, n_samples)


def test_cosine_fractional_n_samples_are_rounded():
    with pytest.warns(UserWarning, match="Fractional"):
        result = generator.cosine(5, 100, 5.7)

    assert result.n_samples == 6


def test_cosine_integer_float_n_samples_are_accepted_without_warning():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = generator.cosine(5, 100, 5.0)

    assert result.n_samples == 5
    assert len(recorded) == 0


def test_cosine_fractional_n_samples_rounding_to_zero_raises():
    with pytest.warns(UserWarning, match="Fractional"):
        with pytest.raises(ValueError):
            generator.cosine(5, 100, 0.4)


# --------------------------------------------------
# Amplitude validation
# --------------------------------------------------

@pytest.mark.parametrize("amplitude", [
    True,
    False,
    np.bool_(True),
    "2",
    1 + 2j,
])
def test_cosine_invalid_amplitude_type(amplitude):
    with pytest.raises(TypeError):
        generator.cosine(5, 100, 10, amplitude=amplitude)


@pytest.mark.parametrize("amplitude", [
    np.nan,
    np.inf,
    -np.inf,
])
def test_cosine_non_finite_amplitude(amplitude):
    with pytest.raises(ValueError):
        generator.cosine(5, 100, 10, amplitude=amplitude)


def test_cosine_zero_amplitude_is_allowed():
    result = generator.cosine(5, 100, 10, amplitude=0, offset=3)

    np.testing.assert_allclose(result.data, np.full(10, 3.0))


def test_cosine_negative_amplitude_is_allowed():
    result = generator.cosine(5, 20, 4, amplitude=-1)

    expected = -np.cos(2 * np.pi * 5 * np.arange(4) / 20)

    np.testing.assert_allclose(result.data, expected)


# --------------------------------------------------
# Phase validation
# --------------------------------------------------

@pytest.mark.parametrize("phase", [
    True,
    False,
    np.bool_(True),
    "1.5",
    1 + 2j,
])
def test_cosine_invalid_phase_type(phase):
    with pytest.raises(TypeError):
        generator.cosine(5, 100, 10, phase=phase)


@pytest.mark.parametrize("phase", [
    np.nan,
    np.inf,
    -np.inf,
])
def test_cosine_non_finite_phase(phase):
    with pytest.raises(ValueError):
        generator.cosine(5, 100, 10, phase=phase)


def test_cosine_negative_phase_is_allowed():
    result = generator.cosine(
        0, 100, 5,
        amplitude=2,
        phase=-np.pi / 2,
    )

    np.testing.assert_allclose(result.data, np.zeros(5), atol=1e-12)


def test_cosine_phase_greater_than_two_pi_is_allowed():
    result = generator.cosine(
        0, 100, 5,
        amplitude=2,
        phase=2 * np.pi,
    )

    np.testing.assert_allclose(result.data, np.full(5, 2.0))


# --------------------------------------------------
# Offset validation
# --------------------------------------------------

@pytest.mark.parametrize("offset", [
    True,
    False,
    np.bool_(True),
    "3",
    1 + 2j,
])
def test_cosine_invalid_offset_type(offset):
    with pytest.raises(TypeError):
        generator.cosine(5, 100, 10, offset=offset)


@pytest.mark.parametrize("offset", [
    np.nan,
    np.inf,
    -np.inf,
])
def test_cosine_non_finite_offset(offset):
    with pytest.raises(ValueError):
        generator.cosine(5, 100, 10, offset=offset)


def test_cosine_negative_offset_is_allowed():
    result = generator.cosine(0, 100, 5, offset=-3)

    np.testing.assert_allclose(result.data, np.full(5, -2.0))
