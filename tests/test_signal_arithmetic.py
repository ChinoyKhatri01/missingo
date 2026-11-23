import numpy as np
from missingo import Signal
import pytest
# ============================================================
# ADDITION TESTS
# ============================================================

def test_signal_add_signal():
    sig1 = Signal([1, 2, 3], 4)
    sig2 = Signal([4, 5, 6], 4)

    result = sig1 + sig2

    np.testing.assert_array_equal(result.data, [5, 7, 9])
    assert result.fs == 4


def test_signal_add_scalar():
    sig = Signal([1, 2, 3], 4)

    result = sig + 5

    np.testing.assert_array_equal(result.data, [6, 7, 8])


def test_scalar_add_signal():
    sig = Signal([1, 2, 3], 4)

    result = 5 + sig

    np.testing.assert_array_equal(result.data, [6, 7, 8])


# ============================================================
# SUBTRACTION TESTS
# ============================================================

def test_signal_sub_signal():
    sig1 = Signal([5, 7, 9], 4)
    sig2 = Signal([1, 2, 3], 4)

    result = sig1 - sig2

    np.testing.assert_array_equal(result.data, [4, 5, 6])


def test_signal_sub_scalar():
    sig = Signal([5, 7, 9], 4)

    result = sig - 2

    np.testing.assert_array_equal(result.data, [3, 5, 7])


def test_scalar_sub_signal():
    sig = Signal([1, 2, 3], 4)

    result = 10 - sig

    np.testing.assert_array_equal(result.data, [9, 8, 7])


# ============================================================
# MULTIPLICATION TESTS
# ============================================================

def test_signal_mul_signal():
    sig1 = Signal([1, 2, 3], 4)
    sig2 = Signal([4, 5, 6], 4)

    result = sig1 * sig2

    np.testing.assert_array_equal(result.data, [4, 10, 18])


def test_signal_mul_scalar():
    sig = Signal([1, 2, 3], 4)

    result = sig * 5

    np.testing.assert_array_equal(result.data, [5, 10, 15])


def test_scalar_mul_signal():
    sig = Signal([1, 2, 3], 4)

    result = 5 * sig

    np.testing.assert_array_equal(result.data, [5, 10, 15])


# ============================================================
# DIVISION TESTS
# ============================================================

def test_signal_div_signal():
    sig1 = Signal([10, 20, 30], 4)
    sig2 = Signal([2, 4, 5], 4)

    result = sig1 / sig2

    np.testing.assert_allclose(result.data, [5, 5, 6])


def test_signal_div_scalar():
    sig = Signal([10, 20, 30], 4)

    result = sig / 10

    np.testing.assert_allclose(result.data, [1, 2, 3])


def test_scalar_div_signal():
    sig = Signal([10, 20, 30], 4)

    result = 60 / sig

    np.testing.assert_allclose(result.data, [6, 3, 2])


# ============================================================
# DIVISION BY ZERO TESTS
# ============================================================

def test_signal_div_zero_scalar():
    sig = Signal([1, 2, 3], 4)

    with pytest.raises(ZeroDivisionError):
        sig / 0


def test_signal_div_numpy_zero_scalar():
    sig = Signal([1, 2, 3], 4)

    with pytest.raises(ZeroDivisionError):
        sig / np.float64(0)


def test_signal_div_signal_containing_zero():
    sig1 = Signal([1, 2, 3], 4)
    sig2 = Signal([1, 0, 3], 4)

    with pytest.raises(ZeroDivisionError):
        sig1 / sig2


def test_scalar_div_signal_containing_zero():
    sig = Signal([1, 0, 3], 4)

    with pytest.raises(ZeroDivisionError):
        10 / sig


# ============================================================
# SAMPLING FREQUENCY AND LENGTH VALIDATION
# ============================================================

@pytest.mark.parametrize(
    "operation",
    [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
    ]
)
def test_arithmetic_with_different_fs(operation):
    sig1 = Signal([1, 2, 3], 4)
    sig2 = Signal([1, 2, 3], 8)

    with pytest.raises(ValueError):
        operation(sig1, sig2)


@pytest.mark.parametrize(
    "operation",
    [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
    ]
)
def test_arithmetic_with_different_lengths(operation):
    sig1 = Signal([1, 2, 3], 4)
    sig2 = Signal([1, 2], 4)

    with pytest.raises(ValueError):
        operation(sig1, sig2)


# ============================================================
# INVALID OPERAND VALIDATION
# ============================================================

@pytest.mark.parametrize(
    "invalid_input",
    [
        True,
        False,
        np.bool_(True),
        "hello",
        [1, 2, 3],
        (1, 2, 3),
        np.array([1, 2, 3]),
        None,
    ]
)
@pytest.mark.parametrize(
    "operation",
    [
        lambda sig, x: sig + x,
        lambda sig, x: sig - x,
        lambda sig, x: sig * x,
        lambda sig, x: sig / x,
    ]
)
def test_invalid_arithmetic_operands(operation, invalid_input):
    sig = Signal([1, 2, 3], 4)

    with pytest.raises(TypeError):
        operation(sig, invalid_input)


# ============================================================
# NON-FINITE SCALAR VALIDATION
# ============================================================

@pytest.mark.parametrize(
    "invalid_scalar",
    [
        np.inf,
        -np.inf,
        np.nan,
        complex(np.inf, 0),
        complex(0, np.nan),
    ]
)
@pytest.mark.parametrize(
    "operation",
    [
        lambda sig, x: sig + x,
        lambda sig, x: sig - x,
        lambda sig, x: sig * x,
        lambda sig, x: sig / x,
    ]
)
def test_non_finite_arithmetic_scalars(operation, invalid_scalar):
    sig = Signal([1, 2, 3], 4)

    with pytest.raises(ValueError):
        operation(sig, invalid_scalar)


# ============================================================
# NUMPY SCALAR SUPPORT
# ============================================================

@pytest.mark.parametrize(
    "scalar",
    [
        np.int64(5),
        np.float64(2.5),
        np.complex128(2 + 3j),
    ]
)
def test_numpy_scalar_addition(scalar):
    sig = Signal([1, 2, 3], 4)

    result = sig + scalar

    np.testing.assert_allclose(result.data, sig.data + scalar)


@pytest.mark.parametrize(
    "scalar",
    [
        np.int64(5),
        np.float64(2.5),
        np.complex128(2 + 3j),
    ]
)
def test_numpy_scalar_multiplication(scalar):
    sig = Signal([1, 2, 3], 4)

    result = sig * scalar

    np.testing.assert_allclose(result.data, sig.data * scalar)


# ============================================================
# COMPLEX NUMBER SUPPORT
# ============================================================

def test_complex_signal_addition():
    sig1 = Signal([1 + 2j, 2 + 3j], 4)
    sig2 = Signal([3 + 4j, 4 + 5j], 4)

    result = sig1 + sig2

    np.testing.assert_allclose(
        result.data,
        [4 + 6j, 6 + 8j]
    )


def test_complex_scalar_multiplication():
    sig = Signal([1, 2, 3], 4)

    result = sig * (2 + 3j)

    np.testing.assert_allclose(
        result.data,
        [2 + 3j, 4 + 6j, 6 + 9j]
    )


def test_complex_scalar_division():
    sig = Signal([1, 2, 3], 4)

    result = sig / (1 + 1j)

    np.testing.assert_allclose(
        result.data,
        np.array([1, 2, 3]) / (1 + 1j)
    )


# ============================================================
# INPUT IMMUTABILITY TESTS
# ============================================================

@pytest.mark.parametrize(
    "operation",
    [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
    ]
)
def test_signal_arithmetic_does_not_modify_inputs(operation):
    sig1 = Signal([10, 20, 30], 4)
    sig2 = Signal([2, 4, 5], 4)

    original1 = sig1.data.copy()
    original2 = sig2.data.copy()

    operation(sig1, sig2)

    np.testing.assert_array_equal(sig1.data, original1)
    np.testing.assert_array_equal(sig2.data, original2)


def test_scalar_arithmetic_does_not_modify_signal():
    sig = Signal([1, 2, 3], 4)
    original = sig.data.copy()

    _ = sig + 5
    _ = sig - 5
    _ = sig * 5
    _ = sig / 5
    _ = 5 + sig
    _ = 5 - sig
    _ = 5 * sig
    _ = 5 / sig

    np.testing.assert_array_equal(sig.data, original)


# ============================================================
# RESULTING SIGNAL VALIDATION
# ============================================================

def test_arithmetic_returns_signal_object():
    sig1 = Signal([1, 2, 3], 4)
    sig2 = Signal([4, 5, 6], 4)

    results = [
        sig1 + sig2,
        sig1 - sig2,
        sig1 * sig2,
        sig1 / sig2,
        sig1 + 5,
        5 - sig1,
        sig1 * 5,
        5 / sig1,
    ]

    for result in results:
        assert isinstance(result, Signal)


def test_arithmetic_preserves_sampling_frequency():
    sig1 = Signal([1, 2, 3], 1000)
    sig2 = Signal([4, 5, 6], 1000)

    results = [
        sig1 + sig2,
        sig1 - sig2,
        sig1 * sig2,
        sig1 / sig2,
        sig1 + 5,
        5 - sig1,
        sig1 * 5,
        5 / sig1,
    ]

    for result in results:
        assert result.fs == 1000
