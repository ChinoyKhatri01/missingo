import numpy as np
from missingo import Signal

def pad(signal, left, right):
    
    #validating inputs
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal object")
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        raise TypeError("left and right must be a non negative integer")
    if not isinstance(left, (int, np.integer)) or not isinstance(right, (int, np.integer)):
        raise TypeError("left and right must be a non negative integer")
    if left < 0 or right < 0 :
        raise ValueError("left and right must be greater or equal to zero")
    
    #performing padding
    n = len(signal.data)
    l = n + left + right
    arr = np.zeros(l, dtype=signal.data.dtype)
    arr[left:left + n] = signal.data[:]

    #creating signal using calculated data
    x = Signal(arr, signal.fs)
    return x
    
    