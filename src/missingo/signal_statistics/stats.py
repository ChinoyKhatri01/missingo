import numpy as np 
from missingo import Signal

def mean(signal):
    #validating the input
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal object")

    #calculating mean
    m = np.mean(signal.data).item()

    return m

def median(signal):
    #validating the input
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal object")
    if np.iscomplexobj(signal.data):
        raise TypeError("median is not defined for complex-valued signals")
    
    #calculating median
    m = np.median(signal.data).item()
    
    return m

def maximum(signal):
    #validating inputs
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal object")
    if np.iscomplexobj(signal.data):
        raise TypeError("maximum is not defined for complex-valued signals")

    #calculating maximum
    m = np.max(signal.data).item()

    return m

def minimum(signal):
     #validating inputs
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal object")
    if np.iscomplexobj(signal.data):
        raise TypeError("minimum is not defined for complex-valued signals")

    #calculating minimum
    m = np.min(signal.data).item()

    return m

def peak_to_peak(signal):
    #calculating peak to peak
    m = maximum(signal) - minimum(signal)

    return m

def mean_absolute_value(signal):
    #validating imput
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal object")

    m = np.mean(np.abs(signal.data)).item()

    return m

def rms(signal):
    #validating inputs
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal object")

    m = (np.mean(np.abs(signal.data)**2)).item()

    return m**(1/2)


    