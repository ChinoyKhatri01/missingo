import numpy as np
from missingo import Signal 

def segmentation_by_count(signal, count):

    #validating the inputs
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal object")
    if isinstance(count, (bool, np.bool_)):
        raise TypeError("count must be an integer")
    if not isinstance(count, (int, np.integer)):
        raise TypeError("count must be an integer")
    if count <= 0 :
        raise ValueError("count must be greater than zero")
    if count > len(signal.data):
        raise ValueError("count can't be greater than number of samples")

    #making segments
    arr = signal.data
    segments = np.array_split(arr, count)

    return [Signal(segment, signal.fs) for segment in segments]


def segmentation_by_length(signal, segment_length):

    #validating the inputs
    if not isinstance(signal, Signal):
        raise TypeError("signal must be a Signal object")
    if isinstance(segment_length, (bool, np.bool_)):
        raise TypeError("segment_length must be an integer")
    if not isinstance(segment_length, (int, np.integer)):
        raise TypeError("segment_length must be an integer")
    if segment_length <= 0:
        raise ValueError("segment_length must be greater than zero")

    #making segments
    arr = signal.data
    segments = [
        arr[i:i + segment_length]
        for i in range(0, len(arr), segment_length)
    ]

    return [Signal(segment, signal.fs) for segment in segments]
