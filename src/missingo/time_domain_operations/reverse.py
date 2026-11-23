from missingo import Signal
import numpy as np

def reverse(signal):
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal object")

    x = Signal(np.flip(signal.data), signal.fs)

    return x