from .._validation import validate_offset
from missingo import Signal

def offset_adjustment(signal, offset):
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal object")
    validate_offset(offset)
    
    #scaled signal generation
    x = signal + offset
    
    return x