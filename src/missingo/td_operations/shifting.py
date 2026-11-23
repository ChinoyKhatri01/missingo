import numpy as np
from missingo import Signal
from .._validation import validate_shift

def shift_fixed(signal, shift):
    #validating the inputs
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal object")
    validate_shift(shift)

    #mathematical operations to perform fixed length shifting
    original = signal.data
    arr = np.zeros_like(original)

    n = len(original)

    if abs(shift) >= n:
        return Signal(arr, signal.fs)

    if shift > 0:
        arr[shift:] = original[:-shift]
    elif shift < 0:
        arr[:shift] = original[-shift:]
    else:
        arr = original.copy()

    #creating signal based on calculated data
    x = Signal(arr, signal.fs)

    return x


def shift_circular(signal, shift):
    #validating the inputs
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal object")
    validate_shift(shift)

    #mathematical operations to perform circular shifting
    arr = np.roll(signal.data, shift)

    #creating signal from calculated
    x = Signal(arr, signal.fs) 

    return x

def shift_extended(signal, shift):
    #validating the inputs
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal object")
    validate_shift(shift)

    original = signal.data.copy()

    n = len(original)

    l = n + abs(shift)

    arr = np.zeros(l, dtype=original.dtype)

    if shift > 0:
        arr[shift:] = original[:]

    elif shift < 0:
        arr[:n] = original[:]

    else :
        arr = original

    x = Signal(arr, signal.fs)

    return x


    