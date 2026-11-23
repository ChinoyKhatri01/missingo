
import numpy as np
import pytest

from missingo import Signal
from missingo.time_domain_operations.segmentation import (
    segmentation_by_count,
    segmentation_by_length,
)


# =====================================================
# Tests for segmentation_by_count()
# =====================================================

def test_segmentation_by_count_normal():
    signal = Signal([1, 2, 3, 4, 5, 6, 7], 100)

    result = segmentation_by_count(signal, 3)

    assert len(result) == 3
    np.testing.assert_array_equal(result[0].data, [1, 2, 3])
    np.testing.assert_array_equal(result[1].data, [4, 5])
    np.testing.assert_array_equal(result[2].data, [6, 7])


def test_segmentation_by_count_one_segment():
    signal = Signal([1, 2, 3, 4], 100)

    result = segmentation_by_count(signal, 1)

    assert len(result) == 1
    np.testing.assert_array_equal(result[0].data, [1, 2, 3, 4])


def test_segmentation_by_count_equal_to_samples():
    signal = Signal([1, 2, 3], 100)

    result = segmentation_by_count(signal, 3)

    assert len(result) == 3

    for i, segment in enumerate(result):
        np.testing.assert_array_equal(segment.data, [i + 1])


def test_segmentation_by_count_preserves_fs():
    signal = Signal([1, 2, 3, 4, 5], 200)

    result = segmentation_by_count(signal, 2)

    for segment in result:
        assert segment.fs == signal.fs


def test_segmentation_by_count_complex():
    signal = Signal([1+2j, 3+4j, 5+6j, 7+8j], 100)

    result = segmentation_by_count(signal, 2)

    np.testing.assert_array_equal(
        result[0].data, [1+2j, 3+4j]
    )
    np.testing.assert_array_equal(
        result[1].data, [5+6j, 7+8j]
    )


def test_segmentation_by_count_returns_signal_objects():
    signal = Signal([1, 2, 3, 4], 100)

    result = segmentation_by_count(signal, 2)

    assert all(isinstance(segment, Signal) for segment in result)


def test_segmentation_by_count_returns_new_objects():
    signal = Signal([1, 2, 3, 4], 100)

    result = segmentation_by_count(signal, 2)

    assert all(segment is not signal for segment in result)


def test_segmentation_by_count_original_unchanged():
    signal = Signal([1, 2, 3, 4, 5], 100)
    original = signal.data.copy()

    segmentation_by_count(signal, 2)

    np.testing.assert_array_equal(signal.data, original)


def test_segmentation_by_count_invalid_signal():
    with pytest.raises(TypeError):
        segmentation_by_count([1, 2, 3], 2)


@pytest.mark.parametrize("count", [True, False, np.bool_(True), np.bool_(False)])
def test_segmentation_by_count_boolean(count):
    signal = Signal([1, 2, 3], 100)

    with pytest.raises(TypeError):
        segmentation_by_count(signal, count)


@pytest.mark.parametrize("count", [2.5, 1.0, "2", None])
def test_segmentation_by_count_invalid_type(count):
    signal = Signal([1, 2, 3], 100)

    with pytest.raises(TypeError):
        segmentation_by_count(signal, count)


@pytest.mark.parametrize("count", [0, -1, -5])
def test_segmentation_by_count_non_positive(count):
    signal = Signal([1, 2, 3], 100)

    with pytest.raises(ValueError):
        segmentation_by_count(signal, count)


def test_segmentation_by_count_greater_than_samples():
    signal = Signal([1, 2, 3], 100)

    with pytest.raises(ValueError):
        segmentation_by_count(signal, 4)


def test_segmentation_by_count_numpy_integer():
    signal = Signal([1, 2, 3, 4], 100)

    result = segmentation_by_count(signal, np.int64(2))

    assert len(result) == 2


# =====================================================
# Tests for segmentation_by_length()
# =====================================================

def test_segmentation_by_length_normal():
    signal = Signal([1, 2, 3, 4, 5, 6, 7], 100)

    result = segmentation_by_length(signal, 2)

    assert len(result) == 4
    np.testing.assert_array_equal(result[0].data, [1, 2])
    np.testing.assert_array_equal(result[1].data, [3, 4])
    np.testing.assert_array_equal(result[2].data, [5, 6])
    np.testing.assert_array_equal(result[3].data, [7])


def test_segmentation_by_length_exact_division():
    signal = Signal([1, 2, 3, 4, 5, 6], 100)

    result = segmentation_by_length(signal, 2)

    assert len(result) == 3

    for segment in result:
        assert len(segment.data) == 2


def test_segmentation_by_length_one_segment():
    signal = Signal([1, 2, 3, 4], 100)

    result = segmentation_by_length(signal, 10)

    assert len(result) == 1
    np.testing.assert_array_equal(result[0].data, [1, 2, 3, 4])


def test_segmentation_by_length_equal_to_samples():
    signal = Signal([1, 2, 3], 100)

    result = segmentation_by_length(signal, 1)

    assert len(result) == 3

    for i, segment in enumerate(result):
        np.testing.assert_array_equal(segment.data, [i + 1])


def test_segmentation_by_length_preserves_fs():
    signal = Signal([1, 2, 3, 4, 5], 200)

    result = segmentation_by_length(signal, 2)

    for segment in result:
        assert segment.fs == signal.fs


def test_segmentation_by_length_complex():
    signal = Signal([1+2j, 3+4j, 5+6j, 7+8j], 100)

    result = segmentation_by_length(signal, 2)

    np.testing.assert_array_equal(
        result[0].data, [1+2j, 3+4j]
    )
    np.testing.assert_array_equal(
        result[1].data, [5+6j, 7+8j]
    )


def test_segmentation_by_length_returns_signal_objects():
    signal = Signal([1, 2, 3, 4], 100)

    result = segmentation_by_length(signal, 2)

    assert all(isinstance(segment, Signal) for segment in result)


def test_segmentation_by_length_returns_new_objects():
    signal = Signal([1, 2, 3, 4], 100)

    result = segmentation_by_length(signal, 2)

    assert all(segment is not signal for segment in result)


def test_segmentation_by_length_original_unchanged():
    signal = Signal([1, 2, 3, 4, 5], 100)
    original = signal.data.copy()

    segmentation_by_length(signal, 2)

    np.testing.assert_array_equal(signal.data, original)


def test_segmentation_by_length_invalid_signal():
    with pytest.raises(TypeError):
        segmentation_by_length([1, 2, 3], 2)


@pytest.mark.parametrize(
    "segment_length",
    [True, False, np.bool_(True), np.bool_(False)]
)
def test_segmentation_by_length_boolean(segment_length):
    signal = Signal([1, 2, 3], 100)

    with pytest.raises(TypeError):
        segmentation_by_length(signal, segment_length)


@pytest.mark.parametrize("segment_length", [2.5, 1.0, "2", None])
def test_segmentation_by_length_invalid_type(segment_length):
    signal = Signal([1, 2, 3], 100)

    with pytest.raises(TypeError):
        segmentation_by_length(signal, segment_length)


@pytest.mark.parametrize("segment_length", [0, -1, -5])
def test_segmentation_by_length_non_positive(segment_length):
    signal = Signal([1, 2, 3], 100)

    with pytest.raises(ValueError):
        segmentation_by_length(signal, segment_length)


def test_segmentation_by_length_numpy_integer():
    signal = Signal([1, 2, 3, 4], 100)

    result = segmentation_by_length(signal, np.int64(2))

    assert len(result) == 2
