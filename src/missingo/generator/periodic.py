from ..signal import Signal
from .._validation import validate_inputs_periodic, validate_duty
import numpy as np

# sine signal function
def sine(frequency, fs, n_samples, amplitude=1.0, phase=0.0, offset=0.0):

    #validation for all inputs
    n_samples = validate_inputs_periodic(frequency, fs, n_samples, amplitude, phase, offset)

    # mathematical expression for sine wave
    x = amplitude*np.sin(2*np.pi*frequency*np.arange(n_samples)/fs + phase) + offset

    # create a Signal object with the generated sine wave data and the specified sampling frequency
    signal = Signal(x, fs)

    return signal

# cosine signal function
def cosine(frequency, fs, n_samples, amplitude=1.0, phase=0.0, offset=0.0):

    #validation for all inputs
    n_samples = validate_inputs_periodic(frequency, fs, n_samples, amplitude, phase, offset)

    # mathematical expression for cosine wave
    x = amplitude*np.cos(2*np.pi*frequency*np.arange(n_samples)/fs + phase) + offset

    # create a Signal object with the generated sine wave data and the specified sampling frequency
    signal = Signal(x, fs)
    
    return signal

#square signal function
def square(frequency, fs, n_samples, amplitude=1.0, phase=0.0, offset=0.0, duty=0.5):

    #validation for all inputs
    n_samples = validate_inputs_periodic(frequency, fs, n_samples, amplitude, phase, offset)

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

#sawtooth signal function
def sawtooth(frequency, fs, n_samples, amplitude=1.0, phase=0.0, offset=0.0):
    #validation for all inputs
    n_samples = validate_inputs_periodic(frequency, fs, n_samples, amplitude, phase, offset)

    #calculating position of the sawtooth wave
    t = np.arange(n_samples) / fs
    p = (frequency * t + phase / (2 * np.pi)) % 1

    # mathematical expression for sawtooth wave
    x = amplitude * (2 * p - 1) + offset

    signal = Signal(x, fs)

    return signal

#triangle signal function
def triangle(frequency, fs, n_samples, amplitude=1.0, phase=0.0, offset=0.0):
    #validation for all inputs
    n_samples = validate_inputs_periodic(frequency, fs, n_samples, amplitude, phase, offset)

    #calculating position of the triangle wave
    t = np.arange(n_samples) / fs
    p = (frequency * t + phase / (2 * np.pi)) % 1

    # mathematical expression for triangle wave
    x = amplitude * (1 - 2 * np.abs(2 * p - 1)) + offset
    signal = Signal(x, fs)

    return signal