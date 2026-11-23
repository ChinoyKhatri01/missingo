import numpy as np
import pytest

from missingo import Signal
from missingo.operations.amplitude import amplitude_scaling


# ============================================================
# BASIC AMPLITUDE SCALING TESTS
# ============================================================

def test_amplitude_scaling_positive_factor():
    sig = Signal([1, 2, 3, 4], 1000)
    result = amplitude_scaling(sig, 3)

    np.testing.assert_array_equal(result.data, [3, 6, 9, 12])


def test_amplitude_scaling_negative_factor():
    sig = Signal([2, -3, 4, -5], 1000)
    result = amplitude_scaling(sig, -2)

    np.testing.assert_array_equal(result.data, [-4, 6, -8, 10])


def test_amplitude_scaling_zero_factor():
    sig = Signal([1, 2, 3, 4], 1000)
    result = amplitude_scaling(sig, 0)

    np.testing.assert_array_equal(result.data, [0, 0, 0, 0])


def test_amplitude_scaling_float_factor():
    sig = Signal([2, 4, 6], 1000)
    result = amplitude_scaling(sig, 0.5)

    np.testing.assert_allclose(result.data, [1, 2, 3])


# ============================================================
# COMPLEX SCALING TESTS
# ============================================================

# def test_amplitude_scaling_complex_factor():
#     sig = Signal([1, 2, 3], 1000)
#     result = amplitude_scaling(sig, 2 + 3j)

#     np.testing.assert_allclose(
#         result.data,
#         [2 + 3j, 4 + 6j, 6 + 9j]
#     )


# def test_amplitude_scaling_complex_signal():
#     sig = Signal([1 + 2j, 3 + 4j], 1000)
#     result = amplitude_scaling(sig, 2)

#     np.testing.assert_allclose(
#         result.data,
#         [2 + 4j, 6 + 8j]
#     )


# ============================================================
# NUMPY SCALAR SUPPORT
# ============================================================

@pytest.mark.parametrize(
    "factor",
    [
        np.int64(3),
        np.float64(2.5),
        np.complex128(2 + 3j),
    ]
)
def test_amplitude_scaling_numpy_scalars(factor):
    sig = Signal([1, 2, 3], 1000)
    result = amplitude_scaling(sig, factor)

    np.testing.assert_allclose(result.data, sig.data * factor)


# ============================================================
# INPUT VALIDATION
# ============================================================

@pytest.mark.parametrize(
    "invalid_signal",
    [
        [1, 2, 3],
        (1, 2, 3),
        np.array([1, 2, 3]),
        10,
        None,
        "signal",
    ]
)
def test_amplitude_scaling_invalid_signal(invalid_signal):
    with pytest.raises(TypeError):
        amplitude_scaling(invalid_signal, 2)


@pytest.mark.parametrize(
    "invalid_factor",
    [
        True,
        False,
        np.bool_(True),
        "hello",
        [1, 2],
        (1, 2),
        np.array([1, 2]),
        None,
    ]
)
def test_amplitude_scaling_invalid_factor(invalid_factor):
    sig = Signal([1, 2, 3], 1000)

    with pytest.raises(TypeError):
        amplitude_scaling(sig, invalid_factor)


# ============================================================
# NON-FINITE FACTOR VALIDATION
# ============================================================

# @pytest.mark.parametrize(
#     "factor",
#     [
#         np.inf,
#         -np.inf,
#         np.nan,
#         complex(np.inf, 0),
#         complex(0, np.nan),
#     ]
# )
# def test_amplitude_scaling_non_finite_factor(factor):
#     sig = Signal([1, 2, 3], 1000)

#     with pytest.raises(ValueError):
#         amplitude_scaling(sig, factor)


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def test_amplitude_scaling_returns_signal():
    sig = Signal([1, 2, 3], 1000)
    result = amplitude_scaling(sig, 2)

    assert isinstance(result, Signal)


def test_amplitude_scaling_preserves_fs():
    sig = Signal([1, 2, 3], 8000)
    result = amplitude_scaling(sig, 2)

    assert result.fs == 8000


def test_amplitude_scaling_preserves_sample_count():
    sig = Signal([1, 2, 3, 4, 5], 1000)
    result = amplitude_scaling(sig, 2)

    assert result.n_samples == sig.n_samples


# ============================================================
# INPUT IMMUTABILITY
# ============================================================

def test_amplitude_scaling_does_not_modify_original_signal():
    sig = Signal([1, 2, 3, 4], 1000)
    original = sig.data.copy()

    amplitude_scaling(sig, 3)

    np.testing.assert_array_equal(sig.data, original)


def test_amplitude_scaling_returns_independent_signal():
    sig = Signal([1, 2, 3], 1000)
    result = amplitude_scaling(sig, 2)

    assert result is not sig