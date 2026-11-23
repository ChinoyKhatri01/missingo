from ..signal import Signal
from .._validation import validate_factor

def amplitude_scaling(signal, factor):
    
    #validateing input
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal object")
    validate_factor(factor)

    # sacled signal generation
    x = signal * factor

    return x
    