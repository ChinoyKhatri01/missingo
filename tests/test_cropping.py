
import numpy as np
import pytest

from missingo import Signal
from missingo.operations.cropping import crop_by_indices, crop_by_time


# =====================================================
# crop_by_indices()
# =====================================================

def test_crop_by_indices_basic():
    signal = Signal([10, 20, 30, 40, 50, 60], 10)

    result = crop_by_indices(signal, 1, 4)

    np.testing.assert_array_equal(result.data, [20, 30, 40])


def test_crop_by_indices_start_at_zero():
    signal = Signal([10, 20, 30, 40], 10)

    result = crop_by_indices(signal, 0, 2)

    np.testing.assert_array_equal(result.data, [10, 20])


def test_crop_by_indices_end_at_signal_length():
    signal = Signal([10, 20, 30, 40], 10)

    result = crop_by_indices(signal, 2, 4)

    np.testing.assert_array_equal(result.data, [30, 40])


def test_crop_by_indices_full_signal():
    signal = Signal([10, 20, 30, 40], 10)

    result = crop_by_indices(signal, 0, 4)

    np.testing.assert_array_equal(result.data, signal.data)


def test_crop_by_indices_preserves_fs():
    signal = Signal([10, 20, 30, 40], 25)

    result = crop_by_indices(signal, 1, 3)

    assert result.fs == signal.fs


def test_crop_by_indices_complex_signal():
    signal = Signal([1+2j, 3+4j, 5+6j, 7+8j], 10)

    result = crop_by_indices(signal, 1, 3)

    np.testing.assert_array_equal(result.data, [3+4j, 5+6j])


@pytest.mark.parametrize("start", [-1, -5])
def test_crop_by_indices_negative_start(start):
    signal = Signal([10, 20, 30, 40], 10)

    with pytest.raises(ValueError):
        crop_by_indices(signal, start, 3)


@pytest.mark.parametrize("start,end", [
    (2, 2),
    (3, 2),
    (4, 2),
])
def test_crop_by_indices_invalid_order(start, end):
    signal = Signal([10, 20, 30, 40], 10)

    with pytest.raises(ValueError):
        crop_by_indices(signal, start, end)


def test_crop_by_indices_end_exceeds_length():
    signal = Signal([10, 20, 30, 40], 10)

    with pytest.raises(ValueError):
        crop_by_indices(signal, 1, 5)


@pytest.mark.parametrize("start,end", [
    (True, 3),
    (1, False),
    (1.5, 3),
    (1, 3.5),
    ("1", 3),
    (1, "3"),
])
def test_crop_by_indices_invalid_types(start, end):
    signal = Signal([10, 20, 30, 40], 10)

    with pytest.raises(TypeError):
        crop_by_indices(signal, start, end)


def test_crop_by_indices_invalid_signal():
    with pytest.raises(TypeError):
        crop_by_indices([10, 20, 30], 0, 2)


def test_crop_by_indices_immutability():
    signal = Signal([10, 20, 30, 40], 10)
    original = signal.data.copy()

    crop_by_indices(signal, 1, 3)

    np.testing.assert_array_equal(signal.data, original)


def test_crop_by_indices_returns_new_signal():
    signal = Signal([10, 20, 30, 40], 10)

    result = crop_by_indices(signal, 1, 3)

    assert isinstance(result, Signal)
    assert result is not signal


# =====================================================
# crop_by_time()
# =====================================================

def test_crop_by_time_aligned_boundaries():
    signal = Signal([10, 20, 30, 40, 50, 60], 10)

    result = crop_by_time(signal, 0.1, 0.4)

    np.testing.assert_array_equal(result.data, [20, 30, 40])


def test_crop_by_time_non_aligned_boundaries():
    signal = Signal([10, 20, 30, 40, 50, 60], 10)

    result = crop_by_time(signal, 0.15, 0.45)

    np.testing.assert_array_equal(result.data, [30, 40, 50])


def test_crop_by_time_start_inclusive():
    signal = Signal([10, 20, 30, 40, 50], 10)

    result = crop_by_time(signal, 0.2, 0.5)

    np.testing.assert_array_equal(result.data, [30, 40, 50])


def test_crop_by_time_end_exclusive():
    signal = Signal([10, 20, 30, 40, 50], 10)

    result = crop_by_time(signal, 0.1, 0.3)

    np.testing.assert_array_equal(result.data, [20, 30])


def test_crop_by_time_start_at_zero():
    signal = Signal([10, 20, 30, 40], 10)

    result = crop_by_time(signal, 0, 0.2)

    np.testing.assert_array_equal(result.data, [10, 20])


def test_crop_by_time_end_at_duration():
    signal = Signal([10, 20, 30, 40], 10)

    result = crop_by_time(signal, 0.2, 0.4)

    np.testing.assert_array_equal(result.data, [30, 40])


def test_crop_by_time_preserves_fs():
    signal = Signal([10, 20, 30, 40], 20)

    result = crop_by_time(signal, 0.05, 0.15)

    assert result.fs == signal.fs


def test_crop_by_time_complex_signal():
    signal = Signal([1+2j, 3+4j, 5+6j, 7+8j], 10)

    result = crop_by_time(signal, 0.1, 0.3)

    np.testing.assert_array_equal(result.data, [3+4j, 5+6j])


@pytest.mark.parametrize("start,end", [
    (0, 0),
    (0.3, 0.2),
    (0.5, 0.5),
])
def test_crop_by_time_invalid_order(start, end):
    signal = Signal([10, 20, 30, 40, 50], 10)

    with pytest.raises(ValueError):
        crop_by_time(signal, start, end)


def test_crop_by_time_negative_start():
    signal = Signal([10, 20, 30, 40], 10)

    with pytest.raises(ValueError):
        crop_by_time(signal, -0.1, 0.2)


def test_crop_by_time_end_exceeds_duration():
    signal = Signal([10, 20, 30, 40], 10)

    with pytest.raises(ValueError):
        crop_by_time(signal, 0.1, 0.5)


@pytest.mark.parametrize("start,end", [
    (np.nan, 0.2),
    (0.1, np.nan),
    (np.inf, 0.2),
    (0.1, np.inf),
    (-np.inf, 0.2),
])
def test_crop_by_time_non_finite_values(start, end):
    signal = Signal([10, 20, 30, 40], 10)

    with pytest.raises(ValueError):
        crop_by_time(signal, start, end)


@pytest.mark.parametrize("start,end", [
    (True, 0.2),
    (0.1, False),
    ("0.1", 0.2),
    (0.1, "0.2"),
    (None, 0.2),
    (0.1, None),
])
def test_crop_by_time_invalid_types(start, end):
    signal = Signal([10, 20, 30, 40], 10)

    with pytest.raises(TypeError):
        crop_by_time(signal, start, end)


def test_crop_by_time_invalid_signal():
    with pytest.raises(TypeError):
        crop_by_time([10, 20, 30], 0.1, 0.2)


def test_crop_by_time_empty_result():
    signal = Signal([10, 20, 30, 40, 50], 10)

    # No sample timestamp lies in [0.15, 0.19)
    with pytest.raises(ValueError):
        crop_by_time(signal, 0.15, 0.19)


def test_crop_by_time_immutability():
    signal = Signal([10, 20, 30, 40], 10)
    original = signal.data.copy()

    crop_by_time(signal, 0.1, 0.3)

    np.testing.assert_array_equal(signal.data, original)


def test_crop_by_time_returns_new_signal():
    signal = Signal([10, 20, 30, 40], 10)

    result = crop_by_time(signal, 0.1, 0.3)

    assert isinstance(result, Signal)
    assert result is not signal
