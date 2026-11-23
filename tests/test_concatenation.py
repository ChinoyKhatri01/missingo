
import numpy as np
import pytest

from missingo import Signal
from missingo.operations.concatenation import concatenate


# =====================================================
# Normal concatenation
# =====================================================

def test_concatenate_two_signals():
    s1 = Signal([1, 2, 3], 100)
    s2 = Signal([4, 5, 6], 100)

    result = concatenate(s1, s2)

    np.testing.assert_array_equal(
        result.data, [1, 2, 3, 4, 5, 6]
    )


def test_concatenate_multiple_signals():
    s1 = Signal([1, 2], 100)
    s2 = Signal([3, 4], 100)
    s3 = Signal([5, 6], 100)
    s4 = Signal([7, 8], 100)

    result = concatenate(s1, s2, s3, s4)

    np.testing.assert_array_equal(
        result.data, [1, 2, 3, 4, 5, 6, 7, 8]
    )


def test_concatenate_different_lengths():
    s1 = Signal([1, 2, 3], 100)
    s2 = Signal([4], 100)
    s3 = Signal([5, 6, 7, 8], 100)

    result = concatenate(s1, s2, s3)

    np.testing.assert_array_equal(
        result.data, [1, 2, 3, 4, 5, 6, 7, 8]
    )


def test_concatenate_preserves_order():
    s1 = Signal([10, 20], 100)
    s2 = Signal([30, 40], 100)
    s3 = Signal([50, 60], 100)

    result = concatenate(s1, s2, s3)

    np.testing.assert_array_equal(
        result.data, [10, 20, 30, 40, 50, 60]
    )


def test_concatenate_preserves_fs():
    s1 = Signal([1, 2], 200)
    s2 = Signal([3, 4], 200)

    result = concatenate(s1, s2)

    assert result.fs == 200


def test_concatenate_preserves_sample_count():
    s1 = Signal([1, 2, 3], 100)
    s2 = Signal([4, 5], 100)
    s3 = Signal([6, 7, 8, 9], 100)

    result = concatenate(s1, s2, s3)

    assert len(result.data) == 9


# =====================================================
# Complex-valued signals
# =====================================================

def test_concatenate_complex_signals():
    s1 = Signal([1+2j, 3+4j], 100)
    s2 = Signal([5+6j, 7+8j], 100)

    result = concatenate(s1, s2)

    np.testing.assert_array_equal(
        result.data,
        [1+2j, 3+4j, 5+6j, 7+8j]
    )


def test_concatenate_mixed_real_and_complex_signals():
    s1 = Signal([1, 2], 100)
    s2 = Signal([3+4j, 5+6j], 100)

    result = concatenate(s1, s2)

    np.testing.assert_array_equal(
        result.data, [1, 2, 3+4j, 5+6j]
    )

    assert np.iscomplexobj(result.data)


# =====================================================
# Return type and independence
# =====================================================

def test_concatenate_returns_signal_object():
    s1 = Signal([1, 2], 100)
    s2 = Signal([3, 4], 100)

    result = concatenate(s1, s2)

    assert isinstance(result, Signal)


def test_concatenate_returns_new_object():
    s1 = Signal([1, 2], 100)
    s2 = Signal([3, 4], 100)

    result = concatenate(s1, s2)

    assert result is not s1
    assert result is not s2


def test_concatenate_original_signals_unchanged():
    s1 = Signal([1, 2], 100)
    s2 = Signal([3, 4], 100)

    original1 = s1.data.copy()
    original2 = s2.data.copy()

    concatenate(s1, s2)

    np.testing.assert_array_equal(s1.data, original1)
    np.testing.assert_array_equal(s2.data, original2)


# =====================================================
# Invalid number of inputs
# =====================================================

def test_concatenate_no_inputs():
    with pytest.raises(ValueError):
        concatenate()


def test_concatenate_one_input():
    s1 = Signal([1, 2, 3], 100)

    with pytest.raises(ValueError):
        concatenate(s1)


# =====================================================
# Invalid signal inputs
# =====================================================

@pytest.mark.parametrize(
    "invalid_input",
    [
        [1, 2, 3],
        np.array([1, 2, 3]),
        10,
        2.5,
        "signal",
        None,
        True,
    ]
)
def test_concatenate_invalid_signal_object(invalid_input):
    s1 = Signal([1, 2, 3], 100)

    with pytest.raises(TypeError):
        concatenate(s1, invalid_input)


def test_concatenate_all_invalid_inputs():
    with pytest.raises(TypeError):
        concatenate([1, 2], [3, 4])


# =====================================================
# Sampling frequency validation
# =====================================================

def test_concatenate_different_fs():
    s1 = Signal([1, 2, 3], 100)
    s2 = Signal([4, 5, 6], 200)

    with pytest.raises(ValueError):
        concatenate(s1, s2)


def test_concatenate_different_fs_among_multiple_signals():
    s1 = Signal([1, 2], 100)
    s2 = Signal([3, 4], 100)
    s3 = Signal([5, 6], 200)

    with pytest.raises(ValueError):
        concatenate(s1, s2, s3)
