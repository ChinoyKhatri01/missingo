
import numpy as np
import pytest

from missingo import Signal
from missingo.operations.padding import pad


# Basic padding behaviour

def test_pad_left():
    signal = Signal([1, 2, 3], 1000)
    result = pad(signal, 2, 0)

    np.testing.assert_array_equal(result.data, [0, 0, 1, 2, 3])


def test_pad_right():
    signal = Signal([1, 2, 3], 1000)
    result = pad(signal, 0, 2)

    np.testing.assert_array_equal(result.data, [1, 2, 3, 0, 0])


def test_pad_both_sides():
    signal = Signal([1, 2, 3], 1000)
    result = pad(signal, 2, 2)

    np.testing.assert_array_equal(result.data, [0, 0, 1, 2, 3, 0, 0])


def test_pad_zero_width():
    signal = Signal([1, 2, 3], 1000)
    result = pad(signal, 0, 0)

    np.testing.assert_array_equal(result.data, [1, 2, 3])


# Signal properties

def test_pad_preserves_sampling_frequency():
    signal = Signal([1, 2, 3], 2000)
    result = pad(signal, 2, 1)

    assert result.fs == signal.fs


def test_pad_updates_sample_count():
    signal = Signal([1, 2, 3], 1000)
    result = pad(signal, 2, 3)

    assert result.n_samples == 8


def test_pad_preserves_complex_data():
    signal = Signal([1 + 2j, 3 + 4j], 1000)
    result = pad(signal, 1, 1)

    np.testing.assert_array_equal(
        result.data,
        [0 + 0j, 1 + 2j, 3 + 4j, 0 + 0j]
    )

    assert np.iscomplexobj(result.data)


def test_pad_returns_signal_object():
    signal = Signal([1, 2, 3], 1000)
    result = pad(signal, 1, 1)

    assert isinstance(result, Signal)


def test_pad_returns_new_object():
    signal = Signal([1, 2, 3], 1000)
    result = pad(signal, 1, 1)

    assert result is not signal


def test_pad_does_not_modify_original():
    signal = Signal([1, 2, 3], 1000)
    original = signal.data.copy()

    pad(signal, 2, 2)

    np.testing.assert_array_equal(signal.data, original)


# Invalid inputs

def test_pad_invalid_signal():
    with pytest.raises(TypeError):
        pad([1, 2, 3], 1, 1)


@pytest.mark.parametrize("left,right", [
    (True, 1),
    (1, False),
    (np.bool_(True), 1),
    (1, np.bool_(False)),
])
def test_pad_rejects_boolean_widths(left, right):
    signal = Signal([1, 2, 3], 1000)

    with pytest.raises(TypeError):
        pad(signal, left, right)


@pytest.mark.parametrize("left,right", [
    (1.5, 1),
    (1, 2.5),
    ("1", 1),
    (1, "2"),
])
def test_pad_rejects_non_integer_widths(left, right):
    signal = Signal([1, 2, 3], 1000)

    with pytest.raises(TypeError):
        pad(signal, left, right)


@pytest.mark.parametrize("left,right", [
    (-1, 0),
    (0, -1),
    (-2, 3),
])
def test_pad_rejects_negative_widths(left, right):
    signal = Signal([1, 2, 3], 1000)

    with pytest.raises(ValueError):
        pad(signal, left, right)


def test_pad_accepts_numpy_integers():
    signal = Signal([1, 2, 3], 1000)
    result = pad(signal, np.int64(1), np.int32(2))

    np.testing.assert_array_equal(result.data, [0, 1, 2, 3, 0, 0])
