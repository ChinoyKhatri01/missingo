import numpy as np
import pytest

from missingo import Signal
from missingo.td_operations.offset_adjustment import offset_adjustment


# ============================================================
# BASIC OFFSET ADJUSTMENT TESTS
# ============================================================

def test_offset_adjustment_positive():
    sig = Signal([1, 2, 3, 4], 1000)

    result = offset_adjustment(sig, 5)

    np.testing.assert_array_equal(result.data, [6, 7, 8, 9])


def test_offset_adjustment_negative():
    sig = Signal([1, 2, 3, 4], 1000)

    result = offset_adjustment(sig, -3)

    np.testing.assert_array_equal(result.data, [-2, -1, 0, 1])


def test_offset_adjustment_zero():
    sig = Signal([1, 2, 3], 1000)

    result = offset_adjustment(sig, 0)

    np.testing.assert_array_equal(result.data, [1, 2, 3])


def test_offset_adjustment_float():
    sig = Signal([1, 2, 3], 1000)

    result = offset_adjustment(sig, 0.5)

    np.testing.assert_allclose(result.data, [1.5, 2.5, 3.5])


# ============================================================
# NUMPY SCALAR SUPPORT
# ============================================================

@pytest.mark.parametrize(
    "offset",
    [
        np.int64(5),
        np.float64(2.5),
    ]
)
def test_offset_adjustment_numpy_scalars(offset):
    sig = Signal([1, 2, 3], 1000)

    result = offset_adjustment(sig, offset)

    np.testing.assert_allclose(result.data, sig.data + offset)


# ============================================================
# COMPLEX SIGNAL SUPPORT
# ============================================================

def test_offset_adjustment_complex_signal():
    sig = Signal([1 + 2j, 3 + 4j], 1000)

    result = offset_adjustment(sig, 2)

    np.testing.assert_allclose(
        result.data,
        [3 + 2j, 5 + 4j]
    )


# ============================================================
# INVALID SIGNAL VALIDATION
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
def test_offset_adjustment_invalid_signal(invalid_signal):
    with pytest.raises(TypeError):
        offset_adjustment(invalid_signal, 2)


# ============================================================
# INVALID OFFSET VALIDATION
# ============================================================

@pytest.mark.parametrize(
    "invalid_offset",
    [
        True,
        False,
        np.bool_(True),
        "hello",
        [1, 2],
        (1, 2),
        np.array([1, 2]),
        None,
        2 + 3j,
        1j,
        np.complex128(2 + 3j),
    ]
)
def test_offset_adjustment_invalid_offset(invalid_offset):
    sig = Signal([1, 2, 3], 1000)

    with pytest.raises(TypeError):
        offset_adjustment(sig, invalid_offset)


# ============================================================
# NON-FINITE OFFSET VALIDATION
# ============================================================

@pytest.mark.parametrize(
    "offset",
    [
        np.inf,
        -np.inf,
        np.nan,
    ]
)
def test_offset_adjustment_non_finite_offset(offset):
    sig = Signal([1, 2, 3], 1000)

    with pytest.raises(ValueError):
        offset_adjustment(sig, offset)


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def test_offset_adjustment_returns_signal():
    sig = Signal([1, 2, 3], 1000)

    result = offset_adjustment(sig, 5)

    assert isinstance(result, Signal)


def test_offset_adjustment_preserves_fs():
    sig = Signal([1, 2, 3], 8000)

    result = offset_adjustment(sig, 5)

    assert result.fs == 8000


def test_offset_adjustment_preserves_sample_count():
    sig = Signal([1, 2, 3, 4, 5], 1000)

    result = offset_adjustment(sig, 5)

    assert result.n_samples == sig.n_samples


# ============================================================
# INPUT IMMUTABILITY
# ============================================================

def test_offset_adjustment_does_not_modify_original_signal():
    sig = Signal([1, 2, 3, 4], 1000)
    original = sig.data.copy()

    offset_adjustment(sig, 5)

    np.testing.assert_array_equal(sig.data, original)


def test_offset_adjustment_returns_independent_signal():
    sig = Signal([1, 2, 3], 1000)

    result = offset_adjustment(sig, 5)

    assert result is not sig