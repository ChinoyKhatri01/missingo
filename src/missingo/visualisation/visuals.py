from missingo import Signal
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot(
    signal,
    style="line",
    color=None,
    linewidth=1.5,
    linestyle="-",
    marker=None,
    markersize=5,
    title=None,
    xlabel="Time (seconds)",
    ylabel="Amplitude",
    grid=True,
    figsize=None,
    dpi=None
):

    # Input validation
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal object")

    if style not in ("line", "stem"):
        raise ValueError("style must be either 'line' or 'stem'")

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Line plot
    if style == "line":

        ax.plot(signal.time, signal.data, color=color, linewidth=linewidth, linestyle=linestyle, marker=marker, markersize=markersize)

    # Stem plot
    elif style == "stem":

        markerline, stemlines, baseline = ax.stem(
            signal.time,
            signal.data
        )

        # Apply customization to stem components
        if color is not None:
            markerline.set_color(color)
            stemlines.set_color(color)
            baseline.set_color(color)

        markerline.set_markersize(markersize)

        if marker is not None:
            markerline.set_marker(marker)

        stemlines.set_linewidth(linewidth)
        stemlines.set_linestyle(linestyle)

    # Configure axes
    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(grid)

    # Return axes for further customization
    return ax


def show(target=None):

    if target is None:
        plt.show()

    elif isinstance(target, Axes):
        plt.figure(target.figure.number)
        plt.show()

    elif isinstance(target, Figure):
        plt.figure(target.number)
        plt.show()

    else:
        raise TypeError(
            "target must be a Figure, Axes, or None"
        )