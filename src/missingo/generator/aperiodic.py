from ..signal import Signal
from .._validation import validate_inputs_aperiodic, validate_delay
import numpy as np

#impulse signal function
def impulse(fs, n_samples, delay=0, amplitude=1.0, offset=0.0):

    #validation for all inputs
    n_samples = validate_inputs_aperiodic(fs, n_samples, amplitude, offset)
    delay = validate_delay(delay, n_samples)

    #mathematical expression for impulse signal
    x = np.full(n_samples, offset, dtype=float)
    x[delay] += amplitude

    # create a Signal object with the generated impulse signal data and the specified sampling frequency
    signal = Signal(x, fs)
    return signal

#step signal function
def step(fs, n_samples, delay=0, amplitude=1.0, offset=0.0):

    #validation for all inputs
    n_samples = validate_inputs_aperiodic(fs, n_samples, amplitude, offset)
    delay = validate_delay(delay, n_samples)

    #mathematical expression for step signal
    x = np.full(n_samples, offset, dtype=float  )
    x[delay:] += amplitude

    # create a Signal object with the generated step signal data and the specified sampling frequency
    signal = Signal(x, fs)
    return signal

#ramp signal function
def ramp(fs, n_samples, delay=0, slope=1.0, offset=0.0):

    #validation for all inputs
    n_samples = validate_inputs_aperiodic(fs, n_samples, slope=slope , offset=offset)
    delay = validate_delay(delay, n_samples)

    n = np.arange(n_samples, dtype = float)
    # unit signal mask
    u = (n >= delay).astype(float)
    #mathematical expression for ramp signal
    x = offset + slope*((n-delay)/fs)*u

    # create a Signal object with the generated ramp signal data and the specified sampling frequency
    signal = Signal(x, fs)

    return signal

#DC signal function
def dc(fs, n_samples, amplitude = 1):

    #validation for all inputs 
    n_samples = validate_inputs_aperiodic(fs, n_samples, amplitude = amplitude)

    #mathematical expression for DC signal
    x = np.full(n_samples, amplitude, dtype = float)

    #create a signal object with the generated DC signal data and the sampling frequency
    signal = Signal(x,fs)

    return signal

#Gaussian pulse function
def gaussian_pulse(fs, n_samples, amplitude=1, mean = 0, std = 1, offset= 0.0):

    #validation for all inputs
    n_samples = validate_inputs_aperiodic(fs, n_samples, amplitude = amplitude, mean = mean, std = std, offset = offset)

    #mathematical expression for gaussian pulse
    t = np.arange(n_samples) / fs
    x = offset + amplitude * np.exp(-((t - mean) ** 2) / (2 * std ** 2))

    #create a signal object with generated Gaussian pulse data and the sampling frequency
    signal = Signal(x,fs)

    return signal 

