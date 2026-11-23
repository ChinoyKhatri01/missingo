import numpy as np 
from missingo import Signal

def concatenate(*signals):

    #validating the inputs
    if len(signals) < 2:
        raise ValueError("at least two signals are required")
    for signal in signals:
        if not isinstance(signal, Signal):
            raise TypeError("all inputs must be Signal objects")
    fs = signals[0].fs
    for signal in signals[1:]:
        if signal.fs != fs:
            raise ValueError("all signals must have the same sampling frequency")

    #concatenating the inputs
    arrays = [signal.data for signal in signals]
    arr = np.concatenate(arrays)

    return Signal(arr, fs)