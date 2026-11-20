from ..signal import Signal
from .._validation import validate_inputs, validate_duty
import numpy as np

# sine function
def sine(frequency, fs, n_samples, amplitude=1.0, phase=0.0, offset=0.0):

    #validation for all inputs
    n_samples = validate_inputs(frequency, fs, n_samples, amplitude, phase, offset)

    # mathematical expression for sine wave
    x = amplitude*np.sin(2*np.pi*frequency*np.arange(n_samples)/fs + phase) + offset

    # create a Signal object with the generated sine wave data and the specified sampling frequency
    signal = Signal(x, fs)

    return signal

# cosine function
def cosine(frequency, fs, n_samples, amplitude=1.0, phase=0.0, offset=0.0):

    #validation for all inputs
    n_samples = validate_inputs(frequency, fs, n_samples, amplitude, phase, offset)

    # mathematical expression for cosine wave
    x = amplitude*np.cos(2*np.pi*frequency*np.arange(n_samples)/fs + phase) + offset

    # create a Signal object with the generated sine wave data and the specified sampling frequency
    signal = Signal(x, fs)
    
    return signal

#square wave function
def square(frequency, fs, n_samples, amplitude=1.0, phase=0.0, offset=0.0, duty=0.5):

    #validation for all inputs
    n_samples = validate_inputs(frequency, fs, n_samples, amplitude, phase, offset)

    #validate duty cycle
    validate_duty(duty)

    #calculating position of the square wave based on the duty cycle
    t = np.arange(n_samples) / fs
    p = (frequency * t + phase / (2 * np.pi)) % 1
    
    # mathematical expression for square wave
    x = np.where(p < duty, amplitude, -amplitude) + offset
    
    # create a Signal object with the generated square wave data and the specified sampling frequency
    signal = Signal(x, fs)
        
    return signal
    
