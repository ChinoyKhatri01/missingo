
import numpy as np
import pytest

from missingo.td_operations.reverse import Signal, reverse


def test_reverse_basic():
    signal = Signal([1, 2, 3, 4, 5], 1000)

    result = reverse(signal)

    np.testing.assert_array_equal(
        result.data,
        [5, 4, 3, 2, 1]
    )


def test_reverse_preserves_fs():
    signal = Signal([1, 2, 3, 4], 2000)

    result = reverse(signal)

    assert result.fs == signal.fs


def test_reverse_preserves_sample_count():
    signal = Signal([1, 2, 3, 4], 1000)

    result = reverse(signal)

    assert result.n_samples == signal.n_samples


def test_reverse_time_axis_starts_at_zero():
    signal = Signal([1, 2, 3, 4], 1000)

    result = reverse(signal)

    expected_time = np.arange(4) / 1000

    np.testing.assert_allclose(result.time, expected_time)


def test_reverse_does_not_modify_original():
    signal = Signal([1, 2, 3, 4], 1000)
    original_data = signal.data.copy()

    reverse(signal)

    np.testing.assert_array_equal(signal.data, original_data)


def test_reverse_returns_new_signal():
    signal = Signal([1, 2, 3], 1000)

    result = reverse(signal)

    assert isinstance(result, Signal)
    assert result is not signal


def test_reverse_twice_restores_original():
    signal = Signal([1, 2, 3, 4, 5], 1000)

    result = reverse(reverse(signal))

    np.testing.assert_array_equal(result.data, signal.data)


def test_reverse_preserves_complex_samples():
    signal = Signal(
        [1 + 2j, 3 + 4j, 5 + 6j],
        1000
    )

    result = reverse(signal)

    np.testing.assert_array_equal(
        result.data,
        [5 + 6j, 3 + 4j, 1 + 2j]
    )


def test_reverse_single_sample_signal():
    signal = Signal([7], 1000)

    result = reverse(signal)

    np.testing.assert_array_equal(result.data, [7])
    assert result.fs == signal.fs
    assert result.n_samples == 1


@pytest.mark.parametrize(
    "invalid_input",
    [
        None,
        [1, 2, 3],
        np.array([1, 2, 3]),
        10,
        "signal"
    ]
)
def test_reverse_invalid_input(invalid_input):
    with pytest.raises(TypeError):
        reverse(invalid_input)
