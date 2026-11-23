
import numpy as np
import pytest

from missingo import Signal
from missingo.operations.shifting import shift_fixed, shift_circular, shift_extended



# ==========================================
# FIXED-LENGTH SHIFTING
# ==========================================

def test_shift_fixed_positive():
    signal = Signal([10, 20, 30, 40, 50], 1000)

    result = shift_fixed(signal, 2)

    np.testing.assert_array_equal(
        result.data, [0, 0, 10, 20, 30]
    )


def test_shift_fixed_negative():
    signal = Signal([10, 20, 30, 40, 50], 1000)

    result = shift_fixed(signal, -2)

    np.testing.assert_array_equal(
        result.data, [30, 40, 50, 0, 0]
    )


def test_shift_fixed_zero():
    signal = Signal([10, 20, 30, 40, 50], 1000)

    result = shift_fixed(signal, 0)

    np.testing.assert_array_equal(result.data, signal.data)


def test_shift_fixed_equal_to_length():
    signal = Signal([1, 2, 3, 4, 5], 1000)

    result = shift_fixed(signal, 5)

    np.testing.assert_array_equal(result.data, [0, 0, 0, 0, 0])


def test_shift_fixed_negative_equal_to_length():
    signal = Signal([1, 2, 3, 4, 5], 1000)

    result = shift_fixed(signal, -5)

    np.testing.assert_array_equal(result.data, [0, 0, 0, 0, 0])


def test_shift_fixed_exceeds_length():
    signal = Signal([1, 2, 3], 1000)

    result = shift_fixed(signal, 7)

    np.testing.assert_array_equal(result.data, [0, 0, 0])


def test_shift_fixed_complex():
    signal = Signal([1+2j, 3+4j, 5+6j], 1000)

    result = shift_fixed(signal, 1)

    np.testing.assert_array_equal(
        result.data, [0+0j, 1+2j, 3+4j]
    )


def test_shift_fixed_preserves_fs():
    signal = Signal([1, 2, 3], 2000)

    result = shift_fixed(signal, 1)

    assert result.fs == signal.fs


def test_shift_fixed_preserves_length():
    signal = Signal([1, 2, 3, 4], 1000)

    result = shift_fixed(signal, 2)

    assert result.n_samples == signal.n_samples


# ==========================================
# CIRCULAR SHIFTING
# ==========================================

def test_shift_circular_positive():
    signal = Signal([10, 20, 30, 40, 50], 1000)

    result = shift_circular(signal, 2)

    np.testing.assert_array_equal(
        result.data, [40, 50, 10, 20, 30]
    )


def test_shift_circular_negative():
    signal = Signal([10, 20, 30, 40, 50], 1000)

    result = shift_circular(signal, -2)

    np.testing.assert_array_equal(
        result.data, [30, 40, 50, 10, 20]
    )


def test_shift_circular_zero():
    signal = Signal([1, 2, 3], 1000)

    result = shift_circular(signal, 0)

    np.testing.assert_array_equal(result.data, signal.data)


def test_shift_circular_exceeds_length():
    signal = Signal([10, 20, 30, 40, 50], 1000)

    result = shift_circular(signal, 7)

    np.testing.assert_array_equal(
        result.data, [40, 50, 10, 20, 30]
    )


def test_shift_circular_negative_exceeds_length():
    signal = Signal([10, 20, 30, 40, 50], 1000)

    result = shift_circular(signal, -7)

    np.testing.assert_array_equal(
        result.data, [30, 40, 50, 10, 20]
    )


def test_shift_circular_complex():
    signal = Signal([1+2j, 3+4j, 5+6j], 1000)

    result = shift_circular(signal, 1)

    np.testing.assert_array_equal(
        result.data, [5+6j, 1+2j, 3+4j]
    )


def test_shift_circular_preserves_fs():
    signal = Signal([1, 2, 3], 2000)

    result = shift_circular(signal, 2)

    assert result.fs == signal.fs


def test_shift_circular_preserves_length():
    signal = Signal([1, 2, 3, 4], 1000)

    result = shift_circular(signal, 2)

    assert result.n_samples == signal.n_samples


# ==========================================
# EXTENDED SHIFTING
# ==========================================

def test_shift_extended_positive():
    signal = Signal([10, 20, 30, 40, 50], 1000)

    result = shift_extended(signal, 2)

    np.testing.assert_array_equal(
        result.data, [0, 0, 10, 20, 30, 40, 50]
    )


def test_shift_extended_negative():
    signal = Signal([10, 20, 30, 40, 50], 1000)

    result = shift_extended(signal, -2)

    np.testing.assert_array_equal(
        result.data, [10, 20, 30, 40, 50, 0, 0]
    )


def test_shift_extended_zero():
    signal = Signal([10, 20, 30], 1000)

    result = shift_extended(signal, 0)

    np.testing.assert_array_equal(result.data, signal.data)


def test_shift_extended_preserves_original_samples():
    signal = Signal([1, 2, 3, 4], 1000)

    result = shift_extended(signal, 3)

    np.testing.assert_array_equal(
        result.data[3:], signal.data
    )


def test_shift_extended_complex():
    signal = Signal([1+2j, 3+4j, 5+6j], 1000)

    result = shift_extended(signal, 2)

    np.testing.assert_array_equal(
        result.data,
        [0+0j, 0+0j, 1+2j, 3+4j, 5+6j]
    )


def test_shift_extended_increases_length():
    signal = Signal([1, 2, 3, 4], 1000)

    result = shift_extended(signal, 2)

    assert result.n_samples == signal.n_samples + 2


def test_shift_extended_preserves_fs():
    signal = Signal([1, 2, 3], 2000)

    result = shift_extended(signal, 2)

    assert result.fs == signal.fs


# ==========================================
# COMMON VALIDATION AND IMMUTABILITY
# ==========================================

@pytest.mark.parametrize(
    "operation",
    [shift_fixed, shift_circular, shift_extended]
)
@pytest.mark.parametrize(
    "invalid_shift",
    [True, False, 2.5, -1.5, "2", None]
)
def test_shifting_invalid_shift(operation, invalid_shift):
    signal = Signal([1, 2, 3, 4], 1000)

    with pytest.raises(TypeError):
        operation(signal, invalid_shift)


@pytest.mark.parametrize(
    "operation",
    [shift_fixed, shift_circular, shift_extended]
)
@pytest.mark.parametrize(
    "invalid_signal",
    [None, [1, 2, 3], 10, "signal"]
)
def test_shifting_invalid_signal(operation, invalid_signal):
    with pytest.raises(TypeError):
        operation(invalid_signal, 2)


@pytest.mark.parametrize(
    "operation",
    [shift_fixed, shift_circular, shift_extended]
)
def test_shifting_does_not_modify_original(operation):
    signal = Signal([1, 2, 3, 4, 5], 1000)
    original_data = signal.data.copy()

    operation(signal, 2)

    np.testing.assert_array_equal(signal.data, original_data)


@pytest.mark.parametrize(
    "operation",
    [shift_fixed, shift_circular, shift_extended]
)
def test_shifting_returns_new_signal(operation):
    signal = Signal([1, 2, 3, 4, 5], 1000)

    result = operation(signal, 2)

    assert isinstance(result, Signal)
    assert result is not signal
