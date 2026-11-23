import numpy as np 
from missingo import Signal
from .._validation import validate_shift

def crop_by_indices(signal,start, end):

    #validating all innputs
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal object")
   
    if isinstance(start, (bool, np.bool_)):
        raise TypeError("start must be an integer")
    
    if not isinstance(start, (int, np.integer)):
        raise TypeError("start must be an integer")
    
    if isinstance(end, (bool, np.bool_)):
        raise TypeError("end must be an integer")
    
    if not isinstance(end, (int, np.integer)):
        raise TypeError("end must be an integer")

    if start <0 :
        raise ValueError("start must be greater than or equal to 0")

    if start >= end:
        raise ValueError("start must be smaller than end")

    if end > len(signal.data):
        raise ValueError("end must be smaller than or equal to number of samples")
    
    #calculating cropped signal
    original = signal.data
    arr = original[start:end]
    
    #creating signal from generated data
    x = Signal(arr, signal.fs)

    return x

def crop_by_time(signal,start,end):

    #validating input
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a signal object")

    if isinstance(start, (bool, np.bool_)):
        raise TypeError("start must be an integer")
        
    if not isinstance(start, (int, np.integer, float, np.floating)):
        raise TypeError("start must be an integer")
        
    if isinstance(end, (bool, np.bool_)):
        raise TypeError("end must be an integer")
        
    if not isinstance(end, (int, np.integer, float, np.floating)):
        raise TypeError("end must be an integer")

    if not np.isfinite(start) or not np.isfinite(end):
        raise ValueError("time boundaries must be finite")

    if start <0 :
        raise ValueError("start must be greater than or equal to 0")
    
    if start >= end:
        raise ValueError("start must be smaller than end")
    #calculating cropped signal and validating if end is more than duration
    original = signal.data
    n = len(original)
    duration = n / signal.fs
    if end > duration:
        raise ValueError("end must not exceed signal duration")
    time = np.arange(n) / signal.fs
    s = np.searchsorted(time, start, side="left")
    e = np.searchsorted(time, end, side="left")
    arr = original[s:e]

    #createing signal using generated data
    x = Signal(arr, signal.fs)

    return x
