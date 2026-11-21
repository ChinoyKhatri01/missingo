
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






import numpy as np
import pytest

from missingo.generator import square


# --------------------------------------------------
# Basic waveform tests
# --------------------------------------------------

def test_square_default():
    signal = square(
        frequency=1,
        fs=8,
        n_samples=8
    )

    expected = np.array([
        1, 1, 1, 1,
        -1, -1, -1, -1
    ])

    np.testing.assert_array_equal(signal.data, expected)


def test_square_duty_25_percent():
    signal = square(
        frequency=1,
        fs=8,
        n_samples=8,
        duty=0.25
    )

    expected = np.array([
        1, 1,
        -1, -1, -1, -1, -1, -1
    ])

    np.testing.assert_array_equal(signal.data, expected)


def test_square_duty_75_percent():
    signal = generator.square(
        frequency=1,
        fs=8,
        n_samples=8,
        duty=0.75
    )

    expected = np.array([
        1, 1, 1, 1, 1, 1,
        -1, -1
    ])

    np.testing.assert_array_equal(signal.data, expected)


# --------------------------------------------------
# Amplitude and offset tests
# --------------------------------------------------

def test_square_custom_amplitude():
    signal = generator.square(
        frequency=1,
        fs=8,
        n_samples=8,
        amplitude=2
    )

    expected = np.array([
        2, 2, 2, 2,
        -2, -2, -2, -2
    ])

    np.testing.assert_array_equal(signal.data, expected)


def test_square_custom_offset():
    signal = generator.square(
        frequency=1,
        fs=8,
        n_samples=8,
        offset=3
    )

    expected = np.array([
        4, 4, 4, 4,
        2, 2, 2, 2
    ])

    np.testing.assert_array_equal(signal.data, expected)


def test_square_custom_amplitude_and_offset():
    signal = generator.square(
        frequency=1,
        fs=8,
        n_samples=8,
        amplitude=2,
        offset=3
    )

    expected = np.array([
        5, 5, 5, 5,
        1, 1, 1, 1
    ])

    np.testing.assert_array_equal(signal.data, expected)


# --------------------------------------------------
# Phase tests
# --------------------------------------------------

def test_square_phase():
    signal = generator.square(
        frequency=1,
        fs=8,
        n_samples=8,
        phase=np.pi / 2
    )

    expected = np.array([
        1, 1,
        -1, -1, -1, -1,
        1, 1
    ])

    np.testing.assert_array_equal(signal.data, expected)


def test_square_negative_phase():
    signal = generator.square(
        frequency=1,
        fs=8,
        n_samples=8,
        phase=-np.pi / 2
    )

    expected = np.array([
    -1, -1,
     1,  1,  1,  1,
    -1, -1
])

    np.testing.assert_array_equal(signal.data, expected)


# --------------------------------------------------
# Frequency tests
# --------------------------------------------------

def test_square_zero_frequency():
    signal = generator.square(
        frequency=0,
        fs=8,
        n_samples=8
    )

    expected = np.ones(8)

    np.testing.assert_array_equal(signal.data, expected)


# --------------------------------------------------
# Duty cycle validation tests
# --------------------------------------------------

@pytest.mark.parametrize("duty", [
    0,
    1,
    -0.1,
    1.1,
    np.nan,
    np.inf,
    -np.inf
])
def test_square_invalid_duty_values(duty):
    with pytest.raises(ValueError):
        generator.square(
            frequency=1,
            fs=8,
            n_samples=8,
            duty=duty
        )


@pytest.mark.parametrize("duty", [
    True,
    False,
    "0.5",
    None,
    [0.5]
])
def test_square_invalid_duty_types(duty):
    with pytest.raises(TypeError):
        generator.square(
            frequency=1,
            fs=8,
            n_samples=8,
            duty=duty
        )


# --------------------------------------------------
# Output tests
# --------------------------------------------------

def test_square_output_length():
    signal = generator.square(
        frequency=1,
        fs=100,
        n_samples=500
    )

    assert len(signal.data) == 500


def test_square_output_is_finite():
    signal = generator.square(
        frequency=1,
        fs=100,
        n_samples=500
    )

    assert np.all(np.isfinite(signal.data))


# ==================================================
# DAY 7: SAWTOOTH, TRIANGLE, IMPULSE AND STEP TESTS
# ==================================================


# --------------------------------------------------
# SAWTOOTH GENERATOR TESTS
# --------------------------------------------------

@pytest.mark.parametrize("waveform", [
    generator.sawtooth,
    generator.triangle,
])
def test_day7_periodic_returns_signal(waveform):
    result = waveform(1, 8, 8)
    assert isinstance(result, Signal)


def test_sawtooth_default_values():
    result = generator.sawtooth(1, 8, 8)

    expected = np.array([
        -1, -0.75, -0.5, -0.25,
         0,  0.25,  0.5,  0.75
    ])

    np.testing.assert_allclose(result.data, expected)


def test_sawtooth_custom_amplitude():
    result = generator.sawtooth(1, 8, 8, amplitude=2)

    expected = np.array([
        -2, -1.5, -1, -0.5,
         0,  0.5,  1,  1.5
    ])

    np.testing.assert_allclose(result.data, expected)


def test_sawtooth_negative_amplitude():
    result = generator.sawtooth(1, 8, 8, amplitude=-2)

    expected = np.array([
         2, 1.5, 1, 0.5,
         0, -0.5, -1, -1.5
    ])

    np.testing.assert_allclose(result.data, expected)


def test_sawtooth_custom_offset():
    result = generator.sawtooth(1, 8, 8, offset=3)

    expected = np.array([
        2, 2.25, 2.5, 2.75,
        3, 3.25, 3.5, 3.75
    ])

    np.testing.assert_allclose(result.data, expected)


def test_sawtooth_custom_amplitude_offset():
    result = generator.sawtooth(
        1, 8, 8,
        amplitude=2,
        offset=3
    )

    expected = np.array([
        1, 1.5, 2, 2.5,
        3, 3.5, 4, 4.5
    ])

    np.testing.assert_allclose(result.data, expected)


def test_sawtooth_phase():
    result = generator.sawtooth(
        1, 8, 8,
        phase=np.pi / 2
    )

    n = np.arange(8)
    p = (n / 8 + 0.25) % 1
    expected = 2 * p - 1

    np.testing.assert_allclose(result.data, expected)


def test_sawtooth_zero_frequency():
    result = generator.sawtooth(
        0, 100, 8,
        amplitude=2,
        offset=3
    )

    np.testing.assert_allclose(result.data, np.full(8, 1.0))


# --------------------------------------------------
# TRIANGLE GENERATOR TESTS
# --------------------------------------------------

def test_triangle_default_values():
    result = generator.triangle(1, 8, 8)

    expected = np.array([
        -1, -0.5, 0, 0.5,
         1,  0.5, 0, -0.5
    ])

    np.testing.assert_allclose(result.data, expected)


def test_triangle_custom_amplitude():
    result = generator.triangle(1, 8, 8, amplitude=2)

    expected = np.array([
        -2, -1, 0, 1,
         2,  1, 0, -1
    ])

    np.testing.assert_allclose(result.data, expected)


def test_triangle_negative_amplitude():
    result = generator.triangle(1, 8, 8, amplitude=-2)

    expected = np.array([
         2, 1, 0, -1,
        -2, -1, 0, 1
    ])

    np.testing.assert_allclose(result.data, expected)


def test_triangle_custom_offset():
    result = generator.triangle(1, 8, 8, offset=3)

    expected = np.array([
        2, 2.5, 3, 3.5,
        4, 3.5, 3, 2.5
    ])

    np.testing.assert_allclose(result.data, expected)


def test_triangle_custom_amplitude_offset():
    result = generator.triangle(
        1, 8, 8,
        amplitude=2,
        offset=3
    )

    expected = np.array([
        1, 2, 3, 4,
        5, 4, 3, 2
    ])

    np.testing.assert_allclose(result.data, expected)


def test_triangle_phase():
    result = generator.triangle(
        1, 8, 8,
        phase=np.pi / 2
    )

    n = np.arange(8)
    p = (n / 8 + 0.25) % 1
    expected = 1 - 2 * np.abs(2 * p - 1)

    np.testing.assert_allclose(result.data, expected)


def test_triangle_zero_frequency():
    result = generator.triangle(
        0, 100, 8,
        amplitude=2,
        offset=3
    )

    np.testing.assert_allclose(result.data, np.full(8, 1.0))


# --------------------------------------------------
# PERIODIC GENERATOR VALIDATION
# Applies to sawtooth and triangle
# --------------------------------------------------

@pytest.mark.parametrize("waveform", [
    generator.sawtooth,
    generator.triangle,
])
@pytest.mark.parametrize("frequency", [
    True, False, np.bool_(True),
    "5", 1 + 2j, None
])
def test_day7_periodic_invalid_frequency_type(waveform, frequency):
    with pytest.raises(TypeError):
        waveform(frequency, 100, 10)


@pytest.mark.parametrize("waveform", [
    generator.sawtooth,
    generator.triangle,
])
@pytest.mark.parametrize("frequency", [-1, -0.5])
def test_day7_periodic_negative_frequency(waveform, frequency):
    with pytest.raises(ValueError):
        waveform(frequency, 100, 10)


@pytest.mark.parametrize("waveform", [
    generator.sawtooth,
    generator.triangle,
])
@pytest.mark.parametrize("frequency", [
    np.nan, np.inf, -np.inf
])
def test_day7_periodic_non_finite_frequency(waveform, frequency):
    with pytest.raises(ValueError):
        waveform(frequency, 100, 10)


@pytest.mark.parametrize("waveform", [
    generator.sawtooth,
    generator.triangle,
])
@pytest.mark.parametrize("fs", [
    True, False, np.bool_(True),
    "100", 1 + 2j, None
])
def test_day7_periodic_invalid_fs_type(waveform, fs):
    with pytest.raises(TypeError):
        waveform(5, fs, 10)


@pytest.mark.parametrize("waveform", [
    generator.sawtooth,
    generator.triangle,
])
@pytest.mark.parametrize("fs", [
    0, -100, np.nan, np.inf, -np.inf
])
def test_day7_periodic_invalid_fs_value(waveform, fs):
    with pytest.raises(ValueError):
        waveform(5, fs, 10)


@pytest.mark.parametrize("waveform", [
    generator.sawtooth,
    generator.triangle,
])
@pytest.mark.parametrize("amplitude", [
    True, False, np.bool_(True),
    "2", 1 + 2j, None
])
def test_day7_periodic_invalid_amplitude_type(waveform, amplitude):
    with pytest.raises(TypeError):
        waveform(5, 100, 10, amplitude=amplitude)


@pytest.mark.parametrize("waveform", [
    generator.sawtooth,
    generator.triangle,
])
@pytest.mark.parametrize("amplitude", [
    np.nan, np.inf, -np.inf
])
def test_day7_periodic_non_finite_amplitude(waveform, amplitude):
    with pytest.raises(ValueError):
        waveform(5, 100, 10, amplitude=amplitude)


@pytest.mark.parametrize("waveform", [
    generator.sawtooth,
    generator.triangle,
])
@pytest.mark.parametrize("phase", [
    True, False, np.bool_(True),
    "1.5", 1 + 2j, None
])
def test_day7_periodic_invalid_phase_type(waveform, phase):
    with pytest.raises(TypeError):
        waveform(5, 100, 10, phase=phase)


@pytest.mark.parametrize("waveform", [
    generator.sawtooth,
    generator.triangle,
])
@pytest.mark.parametrize("phase", [
    np.nan, np.inf, -np.inf
])
def test_day7_periodic_non_finite_phase(waveform, phase):
    with pytest.raises(ValueError):
        waveform(5, 100, 10, phase=phase)


@pytest.mark.parametrize("waveform", [
    generator.sawtooth,
    generator.triangle,
])
@pytest.mark.parametrize("offset", [
    True, False, np.bool_(True),
    "3", 1 + 2j, None
])
def test_day7_periodic_invalid_offset_type(waveform, offset):
    with pytest.raises(TypeError):
        waveform(5, 100, 10, offset=offset)


@pytest.mark.parametrize("waveform", [
    generator.sawtooth,
    generator.triangle,
])
@pytest.mark.parametrize("offset", [
    np.nan, np.inf, -np.inf
])
def test_day7_periodic_non_finite_offset(waveform, offset):
    with pytest.raises(ValueError):
        waveform(5, 100, 10, offset=offset)


@pytest.mark.parametrize("waveform", [
    generator.sawtooth,
    generator.triangle,
])
def test_day7_periodic_nyquist_warning(waveform):
    with pytest.warns(UserWarning, match="Nyquist"):
        waveform(60, 100, 10)


@pytest.mark.parametrize("waveform", [
    generator.sawtooth,
    generator.triangle,
])
def test_day7_periodic_fractional_n_samples(waveform):
    with pytest.warns(UserWarning, match="Fractional"):
        result = waveform(5, 100, 5.7)

    assert result.n_samples == 6


# --------------------------------------------------
# IMPULSE GENERATOR TESTS
# --------------------------------------------------

def test_impulse_default_values():
    result = generator.impulse(100, 5)

    expected = np.array([1, 0, 0, 0, 0])

    np.testing.assert_array_equal(result.data, expected)


def test_impulse_custom_delay():
    result = generator.impulse(100, 8, delay=3)

    expected = np.array([0, 0, 0, 1, 0, 0, 0, 0])

    np.testing.assert_array_equal(result.data, expected)


def test_impulse_custom_amplitude():
    result = generator.impulse(
        100, 8,
        delay=3,
        amplitude=5
    )

    expected = np.array([0, 0, 0, 5, 0, 0, 0, 0])

    np.testing.assert_array_equal(result.data, expected)


def test_impulse_negative_amplitude():
    result = generator.impulse(
        100, 8,
        delay=3,
        amplitude=-5
    )

    expected = np.array([0, 0, 0, -5, 0, 0, 0, 0])

    np.testing.assert_array_equal(result.data, expected)


def test_impulse_custom_offset():
    result = generator.impulse(
        100, 8,
        delay=3,
        offset=2
    )

    expected = np.array([2, 2, 2, 3, 2, 2, 2, 2])

    np.testing.assert_array_equal(result.data, expected)


def test_impulse_custom_amplitude_and_offset():
    result = generator.impulse(
        100, 8,
        delay=3,
        amplitude=5,
        offset=2
    )

    expected = np.array([2, 2, 2, 7, 2, 2, 2, 2])

    np.testing.assert_array_equal(result.data, expected)


def test_impulse_at_first_sample():
    result = generator.impulse(100, 5, delay=0, amplitude=4)

    expected = np.array([4, 0, 0, 0, 0])

    np.testing.assert_array_equal(result.data, expected)


def test_impulse_at_last_sample():
    result = generator.impulse(100, 5, delay=4, amplitude=4)

    expected = np.array([0, 0, 0, 0, 4])

    np.testing.assert_array_equal(result.data, expected)


# --------------------------------------------------
# STEP GENERATOR TESTS
# --------------------------------------------------

def test_step_default_values():
    result = generator.step(100, 5)

    expected = np.array([1, 1, 1, 1, 1])

    np.testing.assert_array_equal(result.data, expected)


def test_step_custom_delay():
    result = generator.step(100, 8, delay=3)

    expected = np.array([0, 0, 0, 1, 1, 1, 1, 1])

    np.testing.assert_array_equal(result.data, expected)


def test_step_custom_amplitude():
    result = generator.step(
        100, 8,
        delay=3,
        amplitude=5
    )

    expected = np.array([0, 0, 0, 5, 5, 5, 5, 5])

    np.testing.assert_array_equal(result.data, expected)


def test_step_negative_amplitude():
    result = generator.step(
        100, 8,
        delay=3,
        amplitude=-5
    )

    expected = np.array([0, 0, 0, -5, -5, -5, -5, -5])

    np.testing.assert_array_equal(result.data, expected)


def test_step_custom_offset():
    result = generator.step(
        100, 8,
        delay=3,
        offset=2
    )

    expected = np.array([2, 2, 2, 3, 3, 3, 3, 3])

    np.testing.assert_array_equal(result.data, expected)


def test_step_custom_amplitude_and_offset():
    result = generator.step(
        100, 8,
        delay=3,
        amplitude=5,
        offset=2
    )

    expected = np.array([2, 2, 2, 7, 7, 7, 7, 7])

    np.testing.assert_array_equal(result.data, expected)


def test_step_at_first_sample():
    result = generator.step(100, 5, delay=0, amplitude=4)

    expected = np.array([4, 4, 4, 4, 4])

    np.testing.assert_array_equal(result.data, expected)


def test_step_at_last_sample():
    result = generator.step(100, 5, delay=4, amplitude=4)

    expected = np.array([0, 0, 0, 0, 4])

    np.testing.assert_array_equal(result.data, expected)


# --------------------------------------------------
# APERIODIC GENERATOR VALIDATION
# Applies to impulse and step
# --------------------------------------------------

@pytest.mark.parametrize("waveform", [
    generator.impulse,
    generator.step,
])
@pytest.mark.parametrize("fs", [
    True, False, np.bool_(True),
    "100", 1 + 2j, None
])
def test_day7_aperiodic_invalid_fs_type(waveform, fs):
    with pytest.raises(TypeError):
        waveform(fs, 10)


@pytest.mark.parametrize("waveform", [
    generator.impulse,
    generator.step,
])
@pytest.mark.parametrize("fs", [
    0, -100, np.nan, np.inf, -np.inf
])
def test_day7_aperiodic_invalid_fs_value(waveform, fs):
    with pytest.raises(ValueError):
        waveform(fs, 10)


@pytest.mark.parametrize("waveform", [
    generator.impulse,
    generator.step,
])
@pytest.mark.parametrize("amplitude", [
    True, False, np.bool_(True),
    "2", 1 + 2j, None
])
def test_day7_aperiodic_invalid_amplitude_type(waveform, amplitude):
    with pytest.raises(TypeError):
        waveform(100, 10, amplitude=amplitude)


@pytest.mark.parametrize("waveform", [
    generator.impulse,
    generator.step,
])
@pytest.mark.parametrize("amplitude", [
    np.nan, np.inf, -np.inf
])
def test_day7_aperiodic_non_finite_amplitude(waveform, amplitude):
    with pytest.raises(ValueError):
        waveform(100, 10, amplitude=amplitude)


@pytest.mark.parametrize("waveform", [
    generator.impulse,
    generator.step,
])
@pytest.mark.parametrize("offset", [
    True, False, np.bool_(True),
    "2", 1 + 2j, None
])
def test_day7_aperiodic_invalid_offset_type(waveform, offset):
    with pytest.raises(TypeError):
        waveform(100, 10, offset=offset)


@pytest.mark.parametrize("waveform", [
    generator.impulse,
    generator.step,
])
@pytest.mark.parametrize("offset", [
    np.nan, np.inf, -np.inf
])
def test_day7_aperiodic_non_finite_offset(waveform, offset):
    with pytest.raises(ValueError):
        waveform(100, 10, offset=offset)


@pytest.mark.parametrize("waveform", [
    generator.impulse,
    generator.step,
])
@pytest.mark.parametrize("delay", [
    True, False, np.bool_(True),
     2.5, "2", 1 + 2j, None
])

def test_day7_invalid_delay_type(waveform, delay):
    with pytest.raises(TypeError):
        waveform(100, 10, delay=delay)


@pytest.mark.parametrize("waveform", [
    generator.impulse,
    generator.step,
])
@pytest.mark.parametrize("delay", [-1, 10, 100])
def test_day7_invalid_delay_value(waveform, delay):
    with pytest.raises(ValueError):
        waveform(100, 10, delay=delay)


@pytest.mark.parametrize("waveform", [
    generator.impulse,
    generator.step,
])
def test_day7_numpy_integer_delay_is_accepted(waveform):
    result = waveform(100, 8, delay=np.int64(3))

    assert result.n_samples == 8


@pytest.mark.parametrize("waveform", [
    generator.impulse,
    generator.step,
])
def test_day7_aperiodic_fractional_n_samples(waveform):
    with pytest.warns(UserWarning, match="Fractional"):
        result = waveform(100, 5.7)

    assert result.n_samples == 6


# --------------------------------------------------
# COMMON OUTPUT PROPERTY TESTS
# Applies to all four Day 7 generators
# --------------------------------------------------

@pytest.mark.parametrize("waveform, args", [
    (generator.sawtooth, (1, 100, 100)),
    (generator.triangle, (1, 100, 100)),
    (generator.impulse, (100, 100)),
    (generator.step, (100, 100)),
])
def test_day7_output_is_signal(waveform, args):
    result = waveform(*args)

    assert isinstance(result, Signal)


@pytest.mark.parametrize("waveform, args", [
    (generator.sawtooth, (1, 100, 100)),
    (generator.triangle, (1, 100, 100)),
    (generator.impulse, (100, 100)),
    (generator.step, (100, 100)),
])
def test_day7_output_properties(waveform, args):
    result = waveform(*args)

    assert result.fs == 100.0
    assert result.n_samples == 100
    assert len(result.data) == 100
    assert result.duration == pytest.approx(1.0)
    assert len(result.time) == 100


@pytest.mark.parametrize("waveform, args", [
    (generator.sawtooth, (1, 100, 100)),
    (generator.triangle, (1, 100, 100)),
    (generator.impulse, (100, 100)),
    (generator.step, (100, 100)),
])
def test_day7_output_is_finite(waveform, args):
    result = waveform(*args)

    assert np.all(np.isfinite(result.data))


# --------------------------------------------------
# MATHEMATICAL RELATIONSHIP TESTS
# --------------------------------------------------

@pytest.mark.parametrize("delay", [0, 1, 4, 9])
def test_step_first_difference_equals_impulse(delay):
    n_samples = 10
    amplitude = 5
    offset = 2

    step_signal = generator.step(
        100, n_samples,
        delay=delay,
        amplitude=amplitude,
        offset=offset
    )

    impulse_signal = generator.impulse(
        100, n_samples,
        delay=delay,
        amplitude=amplitude,
        offset=0
    )

    step_difference = np.diff(
        step_signal.data,
        prepend=offset
    )

    np.testing.assert_allclose(
        step_difference,
        impulse_signal.data
    )


def test_impulse_contains_exactly_one_amplitude_change():
    result = generator.impulse(
        100, 10,
        delay=4,
        amplitude=5,
        offset=2
    )

    changed_samples = np.count_nonzero(result.data != 2)

    assert changed_samples == 1


def test_step_remains_constant_after_delay():
    result = generator.step(
        100, 10,
        delay=4,
        amplitude=5,
        offset=2
    )

    assert np.all(result.data[:4] == 2)
    assert np.all(result.data[4:] == 7)


# -----------------------------------
# RAMP MATHEMATICAL BEHAVIOUR
# -----------------------------------

def test_ramp_default():
    signal = generator.ramp(fs=4, n_samples=5)

    expected = np.array([0, 0.25, 0.5, 0.75, 1.0])

    np.testing.assert_allclose(signal.data, expected)


def test_ramp_positive_slope():
    signal = generator.ramp(fs=4, n_samples=5, slope=2)

    expected = np.array([0, 0.5, 1, 1.5, 2])

    np.testing.assert_allclose(signal.data, expected)


def test_ramp_negative_slope():
    signal = generator.ramp(fs=2, n_samples=6, slope=-4)

    expected = np.array([0, -2, -4, -6, -8, -10])

    np.testing.assert_allclose(signal.data, expected)


def test_ramp_zero_slope():
    signal = generator.ramp(fs=4, n_samples=5, slope=0)

    expected = np.zeros(5)

    np.testing.assert_allclose(signal.data, expected)


# -----------------------------------
# DELAY AND OFFSET
# -----------------------------------

def test_ramp_with_delay():
    signal = generator.ramp(fs=4, n_samples=7, delay=2, slope=2)

    expected = np.array([0, 0, 0, 0.5, 1, 1.5, 2])

    np.testing.assert_allclose(signal.data, expected)


def test_ramp_with_offset():
    signal = generator.ramp(fs=4, n_samples=5, offset=3)

    expected = np.array([3, 3.25, 3.5, 3.75, 4])

    np.testing.assert_allclose(signal.data, expected)


def test_ramp_with_delay_and_offset():
    signal = generator.ramp(
        fs=4,
        n_samples=7,
        delay=2,
        slope=2,
        offset=3
    )

    expected = np.array([3, 3, 3, 3.5, 4, 4.5, 5])

    np.testing.assert_allclose(signal.data, expected)


def test_ramp_negative_slope_with_offset_and_delay():
    signal = generator.ramp(
        fs=2,
        n_samples=6,
        delay=2,
        slope=-4,
        offset=5
    )

    expected = np.array([5, 5, 5, 3, 1, -1])

    np.testing.assert_allclose(signal.data, expected)


def test_ramp_with_zero_delay():
    signal = generator.ramp(fs=4, n_samples=4, delay=0, slope=2)

    expected = np.array([0, 0.5, 1, 1.5])

    np.testing.assert_allclose(signal.data, expected)


def test_ramp_accepts_integer_float_delay():
    signal = generator.ramp(fs=4, n_samples=5, delay=2.0)

    expected = np.array([0, 0, 0, 0.25, 0.5])

    np.testing.assert_allclose(signal.data, expected)


# -----------------------------------
# OUTPUT AND SIGNAL METADATA
# -----------------------------------

def test_ramp_returns_signal_object():
    signal = generator.ramp(fs=100, n_samples=100)

    assert isinstance(signal, Signal)


def test_ramp_output_length():
    signal = generator.ramp(fs=100, n_samples=200, delay=50)

    assert signal.n_samples == 200
    assert len(signal.data) == 200


def test_ramp_sampling_frequency():
    signal = generator.ramp(fs=8000, n_samples=100)

    assert signal.fs == 8000


def test_ramp_output_is_finite():
    signal = generator.ramp(
        fs=1000,
        n_samples=100,
        slope=-5,
        offset=10
    )

    assert np.all(np.isfinite(signal.data))


# -----------------------------------
# SLOPE VALIDATION
# -----------------------------------

@pytest.mark.parametrize("slope", [
    True,
    False,
    np.bool_(True),
    "2",
    None,
    2 + 3j
])
def test_ramp_invalid_slope_type(slope):
    with pytest.raises(TypeError):
        generator.ramp(fs=100, n_samples=10, slope=slope)


@pytest.mark.parametrize("slope", [
    np.nan,
    np.inf,
    -np.inf
])
def test_ramp_non_finite_slope(slope):
    with pytest.raises(ValueError):
        generator.ramp(fs=100, n_samples=10, slope=slope)


# -----------------------------------
# OFFSET VALIDATION
# -----------------------------------

@pytest.mark.parametrize("offset", [
    True,
    "5",
    None,
    2 + 3j
])
def test_ramp_invalid_offset_type(offset):
    with pytest.raises(TypeError):
        generator.ramp(fs=100, n_samples=10, offset=offset)


@pytest.mark.parametrize("offset", [
    np.nan,
    np.inf,
    -np.inf
])
def test_ramp_non_finite_offset(offset):
    with pytest.raises(ValueError):
        generator.ramp(fs=100, n_samples=10, offset=offset)


# -----------------------------------
# DELAY VALIDATION
# -----------------------------------

@pytest.mark.parametrize("delay", [
    True,
    False,
    np.bool_(True),
    2.5,
    "2"
])
def test_ramp_invalid_delay(delay):
    with pytest.raises((TypeError, ValueError)):
        generator.ramp(fs=100, n_samples=10, delay=delay)


@pytest.mark.parametrize("delay", [-1, 10])
def test_ramp_delay_out_of_range(delay):
    with pytest.raises(ValueError):
        generator.ramp(fs=100, n_samples=10, delay=delay)


# -----------------------------------
# SAMPLING FREQUENCY VALIDATION
# -----------------------------------

@pytest.mark.parametrize("fs", [
    0,
    -1,
    np.nan,
    np.inf
])
def test_ramp_invalid_fs(fs):
    with pytest.raises((TypeError, ValueError)):
        generator.ramp(fs=fs, n_samples=10)


# -----------------------------------
# SAMPLE COUNT VALIDATION
# -----------------------------------

@pytest.mark.parametrize("n_samples", [
    0,
    -1,
    True,
    "10"
])
def test_ramp_invalid_n_samples(n_samples):
    with pytest.raises((TypeError, ValueError)):
        generator.ramp(fs=100, n_samples=n_samples)

def test_ramp_fractional_n_samples():
    with pytest.warns(UserWarning, match="rounded"):
        signal = generator.ramp(fs=100, n_samples=2.5)

    assert len(signal.data) == 2  # Adjust if your rounding policy produces 3.




# ===================================
# DC SIGNAL TEST SUITE
# ===================================


# -----------------------------------
# MATHEMATICAL BEHAVIOUR
# -----------------------------------

def test_dc_default():
    signal = generator.dc(fs=100, n_samples=5)

    expected = np.array([1, 1, 1, 1, 1])

    np.testing.assert_allclose(signal.data, expected)


def test_dc_positive_amplitude():
    signal = generator.dc(fs=100, n_samples=5, amplitude=5)

    expected = np.full(5, 5.0)

    np.testing.assert_allclose(signal.data, expected)


def test_dc_negative_amplitude():
    signal = generator.dc(fs=100, n_samples=5, amplitude=-3)

    expected = np.full(5, -3.0)

    np.testing.assert_allclose(signal.data, expected)


def test_dc_zero_amplitude():
    signal = generator.dc(fs=100, n_samples=5, amplitude=0)

    expected = np.zeros(5)

    np.testing.assert_allclose(signal.data, expected)


def test_dc_fractional_amplitude():
    signal = generator.dc(fs=100, n_samples=5, amplitude=2.5)

    expected = np.full(5, 2.5)

    np.testing.assert_allclose(signal.data, expected)


# -----------------------------------
# SAMPLING FREQUENCY BEHAVIOUR
# -----------------------------------

def test_dc_different_sampling_frequencies():
    signal_1 = generator.dc(fs=100, n_samples=5, amplitude=4)
    signal_2 = generator.dc(fs=1000, n_samples=5, amplitude=4)

    np.testing.assert_array_equal(signal_1.data, signal_2.data)


# -----------------------------------
# OUTPUT AND SIGNAL METADATA
# -----------------------------------

def test_dc_returns_signal_object():
    signal = generator.dc(fs=100, n_samples=100)

    assert isinstance(signal, Signal)


def test_dc_output_length():
    signal = generator.dc(fs=100, n_samples=200)

    assert signal.n_samples == 200
    assert len(signal.data) == 200


def test_dc_sampling_frequency():
    signal = generator.dc(fs=8000, n_samples=100)

    assert signal.fs == 8000


def test_dc_output_is_finite():
    signal = generator.dc(
        fs=1000,
        n_samples=100,
        amplitude=5
    )

    assert np.all(np.isfinite(signal.data))


def test_dc_output_is_constant():
    signal = generator.dc(
        fs=100,
        n_samples=100,
        amplitude=7
    )

    assert np.all(signal.data == 7)


def test_dc_output_dtype():
    signal = generator.dc(fs=100, n_samples=10)

    assert np.issubdtype(signal.data.dtype, np.floating)


# -----------------------------------
# AMPLITUDE VALIDATION
# -----------------------------------

@pytest.mark.parametrize("amplitude", [
    True,
    False,
    np.bool_(True),
    "5",
    None,
    2 + 3j
])
def test_dc_invalid_amplitude_type(amplitude):
    with pytest.raises(TypeError):
        generator.dc(
            fs=100,
            n_samples=10,
            amplitude=amplitude
        )


@pytest.mark.parametrize("amplitude", [
    np.nan,
    np.inf,
    -np.inf
])
def test_dc_non_finite_amplitude(amplitude):
    with pytest.raises(ValueError):
        generator.dc(
            fs=100,
            n_samples=10,
            amplitude=amplitude
        )


# -----------------------------------
# SAMPLING FREQUENCY VALIDATION
# -----------------------------------

@pytest.mark.parametrize("fs", [
    0,
    -1,
    np.nan,
    np.inf
])
def test_dc_invalid_fs(fs):
    with pytest.raises((TypeError, ValueError)):
        generator.dc(fs=fs, n_samples=10)


# -----------------------------------
# SAMPLE COUNT VALIDATION
# -----------------------------------

@pytest.mark.parametrize("n_samples", [
    0,
    -1,
    True,
    "10"
])
def test_dc_invalid_n_samples(n_samples):
    with pytest.raises((TypeError, ValueError)):
        generator.dc(fs=100, n_samples=n_samples)


def test_dc_fractional_n_samples():
    with pytest.warns(UserWarning, match="rounded"):
        signal = generator.dc(fs=100, n_samples=2.5)

    assert len(signal.data) == 2  # Adjust if your rounding policy produces 3.



# ==========================================
# GAUSSIAN PULSE TEST SUITE
# ==========================================


# ------------------------------------------
# MATHEMATICAL BEHAVIOUR
# ------------------------------------------

def test_gaussian_pulse_default():
    signal = generator.gaussian_pulse(fs=4, n_samples=5)

    t = np.arange(5) / 4
    expected = np.exp(-(t ** 2) / 2)

    np.testing.assert_allclose(signal.data, expected)


def test_gaussian_pulse_positive_amplitude():
    signal = generator.gaussian_pulse(
        fs=4,
        n_samples=5,
        amplitude=5
    )

    t = np.arange(5) / 4
    expected = 5 * np.exp(-(t ** 2) / 2)

    np.testing.assert_allclose(signal.data, expected)


def test_gaussian_pulse_negative_amplitude():
    signal = generator.gaussian_pulse(
        fs=4,
        n_samples=5,
        amplitude=-3
    )

    t = np.arange(5) / 4
    expected = -3 * np.exp(-(t ** 2) / 2)

    np.testing.assert_allclose(signal.data, expected)


def test_gaussian_pulse_zero_amplitude():
    signal = generator.gaussian_pulse(
        fs=4,
        n_samples=5,
        amplitude=0
    )

    expected = np.zeros(5)

    np.testing.assert_allclose(signal.data, expected)


def test_gaussian_pulse_peak_at_mean():
    signal = generator.gaussian_pulse(
        fs=4,
        n_samples=5,
        amplitude=5,
        mean=0.5,
        std=0.25
    )

    assert signal.data[2] == pytest.approx(5.0)


def test_gaussian_pulse_symmetry():
    signal = generator.gaussian_pulse(
        fs=4,
        n_samples=9,
        amplitude=5,
        mean=1.0,
        std=0.5
    )

    np.testing.assert_allclose(
        signal.data,
        signal.data[::-1]
    )


# ------------------------------------------
# MEAN BEHAVIOUR
# ------------------------------------------

def test_gaussian_pulse_with_positive_mean():
    signal = generator.gaussian_pulse(
        fs=4,
        n_samples=5,
        mean=0.5
    )

    assert signal.data[2] == pytest.approx(1.0)


def test_gaussian_pulse_with_negative_mean():
    signal = generator.gaussian_pulse(
        fs=4,
        n_samples=5,
        mean=-0.5
    )

    assert np.all(np.isfinite(signal.data))


def test_gaussian_pulse_with_zero_mean():
    signal = generator.gaussian_pulse(
        fs=4,
        n_samples=5,
        mean=0
    )

    assert signal.data[0] == pytest.approx(1.0)


# ------------------------------------------
# STANDARD DEVIATION BEHAVIOUR
# ------------------------------------------

def test_gaussian_pulse_small_std():
    signal = generator.gaussian_pulse(
        fs=4,
        n_samples=5,
        mean=0.5,
        std=0.25
    )

    assert signal.data[2] == pytest.approx(1.0)


def test_gaussian_pulse_large_std():
    signal = generator.gaussian_pulse(
        fs=4,
        n_samples=5,
        mean=0.5,
        std=1.0
    )

    assert signal.data[1] > 0.0


def test_gaussian_pulse_larger_std_produces_wider_pulse():
    narrow = generator.gaussian_pulse(
        fs=4,
        n_samples=5,
        mean=0.5,
        std=0.25
    )

    wide = generator.gaussian_pulse(
        fs=4,
        n_samples=5,
        mean=0.5,
        std=1.0
    )

    assert wide.data[0] > narrow.data[0]
    assert wide.data[4] > narrow.data[4]


# ------------------------------------------
# OFFSET BEHAVIOUR
# ------------------------------------------

def test_gaussian_pulse_with_positive_offset():
    signal = generator.gaussian_pulse(
        fs=4,
        n_samples=5,
        amplitude=2,
        mean=0,
        std=1,
        offset=3
    )

    expected = 3 + 2 * np.exp(-((np.arange(5) / 4) ** 2) / 2)

    np.testing.assert_allclose(signal.data, expected)


def test_gaussian_pulse_with_negative_offset():
    signal = generator.gaussian_pulse(
        fs=4,
        n_samples=5,
        amplitude=2,
        mean=0,
        std=1,
        offset=-3
    )

    expected = -3 + 2 * np.exp(-((np.arange(5) / 4) ** 2) / 2)

    np.testing.assert_allclose(signal.data, expected)


def test_gaussian_pulse_with_zero_offset():
    signal = generator.gaussian_pulse(
        fs=4,
        n_samples=5,
        offset=0
    )

    expected = np.exp(-((np.arange(5) / 4) ** 2) / 2)

    np.testing.assert_allclose(signal.data, expected)


def test_gaussian_pulse_offset_and_amplitude():
    signal = generator.gaussian_pulse(
        fs=4,
        n_samples=5,
        amplitude=4,
        mean=0.5,
        std=0.25,
        offset=2
    )

    assert signal.data[2] == pytest.approx(6.0)


# ------------------------------------------
# SAMPLING FREQUENCY BEHAVIOUR
# ------------------------------------------

def test_gaussian_pulse_different_sampling_frequencies():
    signal_1 = generator.gaussian_pulse(
        fs=100,
        n_samples=100,
        mean=0.2,
        std=0.1
    )

    signal_2 = generator.gaussian_pulse(
        fs=200,
        n_samples=200,
        mean=0.2,
        std=0.1
    )

    assert signal_1.data[20] == pytest.approx(
        signal_2.data[40]
    )


# ------------------------------------------
# OUTPUT AND SIGNAL METADATA
# ------------------------------------------

def test_gaussian_pulse_returns_signal_object():
    signal = generator.gaussian_pulse(
        fs=100,
        n_samples=100
    )

    assert isinstance(signal, Signal)


def test_gaussian_pulse_output_length():
    signal = generator.gaussian_pulse(
        fs=100,
        n_samples=200
    )

    assert signal.n_samples == 200
    assert len(signal.data) == 200


def test_gaussian_pulse_sampling_frequency():
    signal = generator.gaussian_pulse(
        fs=8000,
        n_samples=100
    )

    assert signal.fs == 8000


def test_gaussian_pulse_output_is_finite():
    signal = generator.gaussian_pulse(
        fs=1000,
        n_samples=100,
        amplitude=5,
        mean=0.05,
        std=0.01,
        offset=2
    )

    assert np.all(np.isfinite(signal.data))


def test_gaussian_pulse_output_dtype():
    signal = generator.gaussian_pulse(
        fs=100,
        n_samples=10
    )

    assert np.issubdtype(signal.data.dtype, np.floating)


# ------------------------------------------
# AMPLITUDE VALIDATION
# ------------------------------------------

@pytest.mark.parametrize("amplitude", [
    True,
    False,
    np.bool_(True),
    "5",
    None,
    2 + 3j
])
def test_gaussian_pulse_invalid_amplitude_type(amplitude):
    with pytest.raises(TypeError):
        generator.gaussian_pulse(
            fs=100,
            n_samples=10,
            amplitude=amplitude
        )


@pytest.mark.parametrize("amplitude", [
    np.nan,
    np.inf,
    -np.inf
])
def test_gaussian_pulse_non_finite_amplitude(amplitude):
    with pytest.raises(ValueError):
        generator.gaussian_pulse(
            fs=100,
            n_samples=10,
            amplitude=amplitude
        )


# ------------------------------------------
# MEAN VALIDATION
# ------------------------------------------

@pytest.mark.parametrize("mean", [
    True,
    False,
    np.bool_(True),
    "2",
    None,
    2 + 3j
])
def test_gaussian_pulse_invalid_mean_type(mean):
    with pytest.raises(TypeError):
        generator.gaussian_pulse(
            fs=100,
            n_samples=10,
            mean=mean
        )


@pytest.mark.parametrize("mean", [
    np.nan,
    np.inf,
    -np.inf
])
def test_gaussian_pulse_non_finite_mean(mean):
    with pytest.raises(ValueError):
        generator.gaussian_pulse(
            fs=100,
            n_samples=10,
            mean=mean
        )


# ------------------------------------------
# STANDARD DEVIATION VALIDATION
# ------------------------------------------

@pytest.mark.parametrize("std", [
    True,
    False,
    np.bool_(True),
    "1",
    None,
    2 + 3j
])
def test_gaussian_pulse_invalid_std_type(std):
    with pytest.raises(TypeError):
        generator.gaussian_pulse(
            fs=100,
            n_samples=10,
            std=std
        )


@pytest.mark.parametrize("std", [
    np.nan,
    np.inf,
    -np.inf
])
def test_gaussian_pulse_non_finite_std(std):
    with pytest.raises(ValueError):
        generator.gaussian_pulse(
            fs=100,
            n_samples=10,
            std=std
        )


@pytest.mark.parametrize("std", [
    0,
    -1,
    -0.5
])
def test_gaussian_pulse_invalid_std_range(std):
    with pytest.raises(ValueError):
        generator.gaussian_pulse(
            fs=100,
            n_samples=10,
            std=std
        )


# ------------------------------------------
# OFFSET VALIDATION
# ------------------------------------------

@pytest.mark.parametrize("offset", [
    True,
    False,
    np.bool_(True),
    "5",
    None,
    2 + 3j
])
def test_gaussian_pulse_invalid_offset_type(offset):
    with pytest.raises(TypeError):
        generator.gaussian_pulse(
            fs=100,
            n_samples=10,
            offset=offset
        )


@pytest.mark.parametrize("offset", [
    np.nan,
    np.inf,
    -np.inf
])
def test_gaussian_pulse_non_finite_offset(offset):
    with pytest.raises(ValueError):
        generator.gaussian_pulse(
            fs=100,
            n_samples=10,
            offset=offset
        )


# ------------------------------------------
# SAMPLING FREQUENCY VALIDATION
# ------------------------------------------

@pytest.mark.parametrize("fs", [
    0,
    -1,
    np.nan,
    np.inf
])
def test_gaussian_pulse_invalid_fs(fs):
    with pytest.raises((TypeError, ValueError)):
        generator.gaussian_pulse(
            fs=fs,
            n_samples=10
        )


# ------------------------------------------
# SAMPLE COUNT VALIDATION
# ------------------------------------------

@pytest.mark.parametrize("n_samples", [
    0,
    -1,
    True,
    "10"
])
def test_gaussian_pulse_invalid_n_samples(n_samples):
    with pytest.raises((TypeError, ValueError)):
        generator.gaussian_pulse(
            fs=100,
            n_samples=n_samples
        )


def test_gaussian_pulse_fractional_n_samples():
    with pytest.warns(UserWarning, match="rounded"):
        signal = generator.gaussian_pulse(
            fs=100,
            n_samples=2.5
        )

    assert len(signal.data) == 2  # Adjust if your rounding policy produces 3.





# ==========================================
# WHITE GAUSSIAN NOISE TEST SUITE
# ==========================================


# MATHEMATICAL AND STATISTICAL BEHAVIOUR

def test_white_noise_gaussian_default():
    signal = generator.white_noise_gaussian(
        fs=1000,
        n_samples=1000,
        seed=42
    )

    assert np.all(np.isfinite(signal.data))


def test_white_noise_gaussian_mean():
    signal = generator.white_noise_gaussian(
        fs=1000,
        n_samples=50000,
        mean=3.0,
        seed=42
    )

    assert np.mean(signal.data) == pytest.approx(3.0, abs=0.03)


def test_white_noise_gaussian_std():
    signal = generator.white_noise_gaussian(
        fs=1000,
        n_samples=50000,
        std=2.0,
        seed=42
    )

    assert np.std(signal.data) == pytest.approx(2.0, abs=0.03)


def test_white_noise_gaussian_variance():
    signal = generator.white_noise_gaussian(
        fs=1000,
        n_samples=50000,
        std=2.0,
        seed=42
    )

    assert np.var(signal.data) == pytest.approx(4.0, abs=0.12)


def test_white_noise_gaussian_negative_mean():
    signal = generator.white_noise_gaussian(
        fs=1000,
        n_samples=50000,
        mean=-3.0,
        seed=42
    )

    assert np.mean(signal.data) == pytest.approx(-3.0, abs=0.03)


# def test_white_noise_gaussian_zero_std():
#     with pytest.raises("std must be greater than zero"):
#         signal = generator.white_noise_gaussian(
#             fs=1000,
#             n_samples=100,
#             std=0.0,
#             seed=42
#         )

    # This should raise ValueError because std must be positive.


# OFFSET BEHAVIOUR

def test_white_noise_gaussian_positive_offset():
    signal = generator.white_noise_gaussian(
        fs=1000,
        n_samples=50000,
        mean=0,
        std=1,
        offset=5,
        seed=42
    )

    assert np.mean(signal.data) == pytest.approx(5.0, abs=0.03)


def test_white_noise_gaussian_negative_offset():
    signal = generator.white_noise_gaussian(
        fs=1000,
        n_samples=50000,
        mean=0,
        std=1,
        offset=-5,
        seed=42
    )

    assert np.mean(signal.data) == pytest.approx(-5.0, abs=0.03)


def test_white_noise_gaussian_mean_and_offset():
    signal = generator.white_noise_gaussian(
        fs=1000,
        n_samples=50000,
        mean=2,
        std=1,
        offset=3,
        seed=42
    )

    assert np.mean(signal.data) == pytest.approx(5.0, abs=0.03)


def test_white_noise_gaussian_offset_does_not_change_std():
    signal = generator.white_noise_gaussian(
        fs=1000,
        n_samples=50000,
        mean=2,
        std=3,
        offset=10,
        seed=42
    )

    assert np.std(signal.data) == pytest.approx(3.0, abs=0.04)


def test_white_noise_gaussian_zero_offset():
    signal = generator.white_noise_gaussian(
        fs=1000,
        n_samples=50000,
        mean=0,
        std=1,
        offset=0,
        seed=42
    )

    assert np.mean(signal.data) == pytest.approx(0.0, abs=0.03)


# RANDOMNESS AND REPRODUCIBILITY

def test_white_noise_gaussian_same_seed_produces_same_output():
    signal_1 = generator.white_noise_gaussian(
        fs=1000,
        n_samples=1000,
        seed=42
    )

    signal_2 = generator.white_noise_gaussian(
        fs=1000,
        n_samples=1000,
        seed=42
    )

    np.testing.assert_array_equal(signal_1.data, signal_2.data)


def test_white_noise_gaussian_different_seeds_produce_different_output():
    signal_1 = generator.white_noise_gaussian(
        fs=1000,
        n_samples=1000,
        seed=42
    )

    signal_2 = generator.white_noise_gaussian(
        fs=1000,
        n_samples=1000,
        seed=100
    )

    assert not np.array_equal(signal_1.data, signal_2.data)


def test_white_noise_gaussian_zero_seed():
    signal = generator.white_noise_gaussian(
        fs=1000,
        n_samples=100,
        seed=0
    )

    assert np.all(np.isfinite(signal.data))


def test_white_noise_gaussian_none_seed():
    signal = generator.white_noise_gaussian(
        fs=1000,
        n_samples=100,
        seed=None
    )

    assert len(signal.data) == 100


# WHITE NOISE PROPERTY

def test_white_noise_gaussian_samples_are_uncorrelated():
    signal = generator.white_noise_gaussian(
        fs=1000,
        n_samples=50000,
        seed=42
    )

    correlation = np.corrcoef(
        signal.data[:-1],
        signal.data[1:]
    )[0, 1]

    assert abs(correlation) < 0.03


# OUTPUT AND METADATA

def test_white_noise_gaussian_returns_signal_object():
    signal = generator.white_noise_gaussian(
        fs=1000,
        n_samples=100
    )

    assert isinstance(signal, Signal)


def test_white_noise_gaussian_output_length():
    signal = generator.white_noise_gaussian(
        fs=1000,
        n_samples=500
    )

    assert signal.n_samples == 500
    assert len(signal.data) == 500


def test_white_noise_gaussian_sampling_frequency():
    signal = generator.white_noise_gaussian(
        fs=8000,
        n_samples=100
    )

    assert signal.fs == 8000


def test_white_noise_gaussian_output_is_finite():
    signal = generator.white_noise_gaussian(
        fs=1000,
        n_samples=1000,
        mean=2,
        std=3,
        offset=5,
        seed=42
    )

    assert np.all(np.isfinite(signal.data))


def test_white_noise_gaussian_output_dtype():
    signal = generator.white_noise_gaussian(
        fs=1000,
        n_samples=100
    )

    assert np.issubdtype(signal.data.dtype, np.floating)


# MEAN VALIDATION

@pytest.mark.parametrize(
    "mean",
    [True, False, np.bool_(True), "2", None, 2 + 3j]
)
def test_white_noise_gaussian_invalid_mean_type(mean):
    with pytest.raises(TypeError):
        generator.white_noise_gaussian(
            fs=1000,
            n_samples=100,
            mean=mean
        )


@pytest.mark.parametrize(
    "mean",
    [np.nan, np.inf, -np.inf]
)
def test_white_noise_gaussian_non_finite_mean(mean):
    with pytest.raises(ValueError):
        generator.white_noise_gaussian(
            fs=1000,
            n_samples=100,
            mean=mean
        )


# STD VALIDATION

@pytest.mark.parametrize(
    "std",
    [True, False, np.bool_(True), "1", None, 2 + 3j]
)
def test_white_noise_gaussian_invalid_std_type(std):
    with pytest.raises(TypeError):
        generator.white_noise_gaussian(
            fs=1000,
            n_samples=100,
            std=std
        )


@pytest.mark.parametrize(
    "std",
    [np.nan, np.inf, -np.inf]
)
def test_white_noise_gaussian_non_finite_std(std):
    with pytest.raises(ValueError):
        generator.white_noise_gaussian(
            fs=1000,
            n_samples=100,
            std=std
        )


@pytest.mark.parametrize(
    "std",
    [0, -1, -0.5]
)
def test_white_noise_gaussian_invalid_std_range(std):
    with pytest.raises(ValueError):
        generator.white_noise_gaussian(
            fs=1000,
            n_samples=100,
            std=std
        )


# OFFSET VALIDATION

@pytest.mark.parametrize(
    "offset",
    [True, False, np.bool_(True), "5", None, 2 + 3j]
)
def test_white_noise_gaussian_invalid_offset_type(offset):
    with pytest.raises(TypeError):
        generator.white_noise_gaussian(
            fs=1000,
            n_samples=100,
            offset=offset
        )


@pytest.mark.parametrize(
    "offset",
    [np.nan, np.inf, -np.inf]
)
def test_white_noise_gaussian_non_finite_offset(offset):
    with pytest.raises(ValueError):
        generator.white_noise_gaussian(
            fs=1000,
            n_samples=100,
            offset=offset
        )


# SEED VALIDATION

@pytest.mark.parametrize(
    "seed",
    [True, False, np.bool_(True), 2.5, "42", 2 + 3j]
)
def test_white_noise_gaussian_invalid_seed_type(seed):
    with pytest.raises(TypeError):
        generator.white_noise_gaussian(
            fs=1000,
            n_samples=100,
            seed=seed
        )


@pytest.mark.parametrize(
    "seed",
    [-1, -10, -100]
)
def test_white_noise_gaussian_negative_seed(seed):
    with pytest.raises(ValueError):
        generator.white_noise_gaussian(
            fs=1000,
            n_samples=100,
            seed=seed
        )


def test_white_noise_gaussian_numpy_integer_seed():
    signal = generator.white_noise_gaussian(
        fs=1000,
        n_samples=100,
        seed=np.int64(42)
    )

    assert len(signal.data) == 100


# FS VALIDATION

@pytest.mark.parametrize(
    "fs",
    [0, -1, np.nan, np.inf]
)
def test_white_noise_gaussian_invalid_fs(fs):
    with pytest.raises((TypeError, ValueError)):
        generator.white_noise_gaussian(
            fs=fs,
            n_samples=100
        )


# N_SAMPLES VALIDATION

@pytest.mark.parametrize(
    "n_samples",
    [0, -1, True, "100"]
)
def test_white_noise_gaussian_invalid_n_samples(n_samples):
    with pytest.raises((TypeError, ValueError)):
        generator.white_noise_gaussian(
            fs=1000,
            n_samples=n_samples
        )


def test_white_noise_gaussian_fractional_n_samples():
    with pytest.warns(UserWarning, match="rounded"):
        signal = generator.white_noise_gaussian(
            fs=1000,
            n_samples=2.5,
            seed=42
        )

    assert len(signal.data) == 2  # Adjust if your rounding policy differs.


# ==========================================
# WHITE UNIFORM NOISE TEST SUITE
# ==========================================


# MATHEMATICAL AND STATISTICAL BEHAVIOUR

def test_white_noise_uniform_default():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=1000,
        seed=42
    )

    assert np.all(np.isfinite(signal.data))


def test_white_noise_uniform_output_within_bounds():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=10000,
        low=-2,
        high=3,
        seed=42
    )

    assert np.all(signal.data >= -2)
    assert np.all(signal.data < 3)


def test_white_noise_uniform_mean():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=50000,
        low=-2,
        high=4,
        seed=42
    )

    expected_mean = (-2 + 4) / 2

    assert np.mean(signal.data) == pytest.approx(
        expected_mean,
        abs=0.03
    )


def test_white_noise_uniform_variance():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=50000,
        low=-2,
        high=4,
        seed=42
    )

    expected_variance = (4 - (-2)) ** 2 / 12

    assert np.var(signal.data) == pytest.approx(
        expected_variance,
        abs=0.04
    )


def test_white_noise_uniform_standard_deviation():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=50000,
        low=-2,
        high=4,
        seed=42
    )

    expected_std = (4 - (-2)) / np.sqrt(12)

    assert np.std(signal.data) == pytest.approx(
        expected_std,
        abs=0.03
    )


def test_white_noise_uniform_positive_bounds():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=10000,
        low=2,
        high=6,
        seed=42
    )

    assert np.all(signal.data >= 2)
    assert np.all(signal.data < 6)


def test_white_noise_uniform_negative_bounds():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=10000,
        low=-6,
        high=-2,
        seed=42
    )

    assert np.all(signal.data >= -6)
    assert np.all(signal.data < -2)


def test_white_noise_uniform_zero_crossing_bounds():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=10000,
        low=-3,
        high=3,
        seed=42
    )

    assert np.all(signal.data >= -3)
    assert np.all(signal.data < 3)


# OFFSET BEHAVIOUR

def test_white_noise_uniform_positive_offset():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=10000,
        low=-1,
        high=1,
        offset=5,
        seed=42
    )

    assert np.all(signal.data >= 4)
    assert np.all(signal.data < 6)


def test_white_noise_uniform_negative_offset():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=10000,
        low=-1,
        high=1,
        offset=-5,
        seed=42
    )

    assert np.all(signal.data >= -6)
    assert np.all(signal.data < -4)


def test_white_noise_uniform_offset_shifts_mean():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=50000,
        low=-2,
        high=4,
        offset=5,
        seed=42
    )

    expected_mean = 5 + (-2 + 4) / 2

    assert np.mean(signal.data) == pytest.approx(
        expected_mean,
        abs=0.03
    )


def test_white_noise_uniform_offset_does_not_change_variance():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=50000,
        low=-2,
        high=4,
        offset=10,
        seed=42
    )

    expected_variance = (4 - (-2)) ** 2 / 12

    assert np.var(signal.data) == pytest.approx(
        expected_variance,
        abs=0.04
    )


def test_white_noise_uniform_zero_offset():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=10000,
        low=-1,
        high=1,
        offset=0,
        seed=42
    )

    assert np.all(signal.data >= -1)
    assert np.all(signal.data < 1)


# RANDOMNESS AND REPRODUCIBILITY

def test_white_noise_uniform_same_seed_produces_same_output():
    signal_1 = generator.white_noise_uniform(
        fs=1000,
        n_samples=1000,
        seed=42
    )

    signal_2 = generator.white_noise_uniform(
        fs=1000,
        n_samples=1000,
        seed=42
    )

    np.testing.assert_array_equal(
        signal_1.data,
        signal_2.data
    )


def test_white_noise_uniform_different_seeds_produce_different_output():
    signal_1 = generator.white_noise_uniform(
        fs=1000,
        n_samples=1000,
        seed=42
    )

    signal_2 = generator.white_noise_uniform(
        fs=1000,
        n_samples=1000,
        seed=100
    )

    assert not np.array_equal(
        signal_1.data,
        signal_2.data
    )


def test_white_noise_uniform_zero_seed():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=100,
        seed=0
    )

    assert len(signal.data) == 100


def test_white_noise_uniform_none_seed():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=100,
        seed=None
    )

    assert len(signal.data) == 100


# WHITE NOISE PROPERTY

def test_white_noise_uniform_samples_are_uncorrelated():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=50000,
        seed=42
    )

    correlation = np.corrcoef(
        signal.data[:-1],
        signal.data[1:]
    )[0, 1]

    assert abs(correlation) < 0.03


# OUTPUT AND METADATA

def test_white_noise_uniform_returns_signal_object():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=100
    )

    assert isinstance(signal, Signal)


def test_white_noise_uniform_output_length():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=500
    )

    assert signal.n_samples == 500
    assert len(signal.data) == 500


def test_white_noise_uniform_sampling_frequency():
    signal = generator.white_noise_uniform(
        fs=8000,
        n_samples=100
    )

    assert signal.fs == 8000


def test_white_noise_uniform_output_is_finite():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=1000,
        low=-5,
        high=5,
        offset=2,
        seed=42
    )

    assert np.all(np.isfinite(signal.data))


def test_white_noise_uniform_output_dtype():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=100
    )

    assert np.issubdtype(signal.data.dtype, np.floating)


# LOW VALIDATION

@pytest.mark.parametrize(
    "low",
    [True, False, np.bool_(True), "2", None, 2 + 3j]
)
def test_white_noise_uniform_invalid_low_type(low):
    with pytest.raises(TypeError):
        generator.white_noise_uniform(
            fs=1000,
            n_samples=100,
            low=low
        )


@pytest.mark.parametrize(
    "low",
    [np.nan, np.inf, -np.inf]
)
def test_white_noise_uniform_non_finite_low(low):
    with pytest.raises(ValueError):
        generator.white_noise_uniform(
            fs=1000,
            n_samples=100,
            low=low
        )


# HIGH VALIDATION

@pytest.mark.parametrize(
    "high",
    [True, False, np.bool_(True), "2", None, 2 + 3j]
)
def test_white_noise_uniform_invalid_high_type(high):
    with pytest.raises(TypeError):
        generator.white_noise_uniform(
            fs=1000,
            n_samples=100,
            high=high
        )


@pytest.mark.parametrize(
    "high",
    [np.nan, np.inf, -np.inf]
)
def test_white_noise_uniform_non_finite_high(high):
    with pytest.raises(ValueError):
        generator.white_noise_uniform(
            fs=1000,
            n_samples=100,
            high=high
        )


# BOUNDARY RELATIONSHIP VALIDATION

@pytest.mark.parametrize(
    "low, high",
    [
        (2, 2),
        (5, 2),
        (10, -10),
        (0, 0)
    ]
)
def test_white_noise_uniform_invalid_bounds(low, high):
    with pytest.raises(ValueError):
        generator.white_noise_uniform(
            fs=1000,
            n_samples=100,
            low=low,
            high=high
        )


# OFFSET VALIDATION

@pytest.mark.parametrize(
    "offset",
    [True, False, np.bool_(True), "5", None, 2 + 3j]
)
def test_white_noise_uniform_invalid_offset_type(offset):
    with pytest.raises(TypeError):
        generator.white_noise_uniform(
            fs=1000,
            n_samples=100,
            offset=offset
        )


@pytest.mark.parametrize(
    "offset",
    [np.nan, np.inf, -np.inf]
)
def test_white_noise_uniform_non_finite_offset(offset):
    with pytest.raises(ValueError):
        generator.white_noise_uniform(
            fs=1000,
            n_samples=100,
            offset=offset
        )


# SEED VALIDATION

@pytest.mark.parametrize(
    "seed",
    [True, False, np.bool_(True), 2.5, "42", 2 + 3j]
)
def test_white_noise_uniform_invalid_seed_type(seed):
    with pytest.raises(TypeError):
        generator.white_noise_uniform(
            fs=1000,
            n_samples=100,
            seed=seed
        )


@pytest.mark.parametrize(
    "seed",
    [-1, -10, -100]
)
def test_white_noise_uniform_negative_seed(seed):
    with pytest.raises(ValueError):
        generator.white_noise_uniform(
            fs=1000,
            n_samples=100,
            seed=seed
        )


def test_white_noise_uniform_numpy_integer_seed():
    signal = generator.white_noise_uniform(
        fs=1000,
        n_samples=100,
        seed=np.int64(42)
    )

    assert len(signal.data) == 100


# FS VALIDATION

@pytest.mark.parametrize(
    "fs",
    [0, -1, np.nan, np.inf]
)
def test_white_noise_uniform_invalid_fs(fs):
    with pytest.raises((TypeError, ValueError)):
        generator.white_noise_uniform(
            fs=fs,
            n_samples=100
        )


# N_SAMPLES VALIDATION

@pytest.mark.parametrize(
    "n_samples",
    [0, -1, True, "100"]
)
def test_white_noise_uniform_invalid_n_samples(n_samples):
    with pytest.raises((TypeError, ValueError)):
        generator.white_noise_uniform(
            fs=1000,
            n_samples=n_samples
        )


def test_white_noise_uniform_fractional_n_samples():
    with pytest.warns(UserWarning, match="rounded"):
        signal = generator.white_noise_uniform(
            fs=1000,
            n_samples=2.5,
            seed=42
        )

    assert len(signal.data) == 2  # Adjust if your rounding policy differs.
