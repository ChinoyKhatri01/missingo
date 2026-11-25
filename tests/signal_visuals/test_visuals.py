
import pytest
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from missingo import Signal
from missingo.visualisation.visuals import plot, show
from matplotlib.axes import Axes
from matplotlib.figure import Figure


@pytest.fixture
def signal():
    return Signal([0, 1, 0, -1], 4)


# -------------------------
# Plot function tests
# -------------------------

def test_plot_line_returns_axes(signal):
    ax = plot(signal)

    assert isinstance(ax, Axes)


def test_plot_stem_returns_axes(signal):
    ax = plot(signal, style="stem")

    assert isinstance(ax, Axes)


def test_plot_returns_axes_with_figure(signal):
    ax = plot(signal)

    assert isinstance(ax.figure, Figure)


def test_plot_invalid_signal():
    with pytest.raises(TypeError):
        plot([1, 2, 3])


def test_plot_invalid_style(signal):
    with pytest.raises(ValueError):
        plot(signal, style="invalid")


# -------------------------
# Plot customization tests
# -------------------------

def test_plot_title(signal):
    ax = plot(signal, title="Test Signal")

    assert ax.get_title() == "Test Signal"


def test_plot_xlabel(signal):
    ax = plot(signal, xlabel="Time")

    assert ax.get_xlabel() == "Time"


def test_plot_ylabel(signal):
    ax = plot(signal, ylabel="Amplitude")

    assert ax.get_ylabel() == "Amplitude"


def test_plot_grid_enabled(signal):
    ax = plot(signal, grid=True)

    assert any(line.get_visible() for line in ax.get_xgridlines())


def test_plot_grid_disabled(signal):
    ax = plot(signal, grid=False)

    assert not any(line.get_visible() for line in ax.get_xgridlines())


def test_plot_figsize(signal):
    ax = plot(signal, figsize=(10, 5))

    assert ax.figure.get_size_inches().tolist() == [10, 5]


def test_plot_dpi(signal):
    ax = plot(signal, dpi=150)

    assert ax.figure.dpi == 150


def test_plot_line_color(signal):
    ax = plot(signal, color="orange")

    assert ax.lines[0].get_color() == "orange"


def test_plot_linewidth(signal):
    ax = plot(signal, linewidth=3)

    assert ax.lines[0].get_linewidth() == 3


def test_plot_linestyle(signal):
    ax = plot(signal, linestyle="--")

    assert ax.lines[0].get_linestyle() == "--"


def test_plot_marker(signal):
    ax = plot(signal, marker="o")

    assert ax.lines[0].get_marker() == "o"


def test_plot_markersize(signal):
    ax = plot(signal, markersize=8, marker="o")

    assert ax.lines[0].get_markersize() == 8


# -------------------------
# Show function tests
# -------------------------

def test_show_all_figures(monkeypatch):
    called = []

    monkeypatch.setattr(plt, "show", lambda: called.append(True))

    show()

    assert called == [True]


def test_show_figure(monkeypatch, signal):
    ax = plot(signal)
    fig = ax.figure

    called = []

    def mock_show():
        called.append(plt.gcf())

    monkeypatch.setattr(plt, "show", mock_show)

    show(fig)

    assert called == [fig]


def test_show_axes(monkeypatch, signal):
    ax = plot(signal)

    called = []

    def mock_show():
        called.append(plt.gcf())

    monkeypatch.setattr(plt, "show", mock_show)

    show(ax)

    assert called == [ax.figure]

def test_show_invalid_target():
    with pytest.raises(TypeError):
        show("invalid")
