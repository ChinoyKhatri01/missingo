from missingo import Signal
from .._validation import validate_amplitude
import warnings
import numpy as np

def normalisation(signal, amplitude=1):
    #input validation
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal object")
    validate_amplitude(amplitude)
    if amplitude < 0 :
        amplitude = abs(amplitude)
        warnings.warn(
                "Normalization requires a non-negative amplitude. "
                "Converting the supplied amplitude to its absolute value.",
                UserWarning
            )
        
    #calculating peak
    peak = np.max(np.abs(signal.data))

    #peak validation
    if peak == 0:
        raise ValueError("Cannot normalize a zero-valued signal")
    
    #generating normalised signal
    x = (signal*amplitude)/peak

    return x