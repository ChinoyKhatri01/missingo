from ..signal import Signal
import numpy as np
import warnings

# sine function

def sine(frequency, fs, n_samples, amplitude=1.0, phase=0.0, offset=0.0):

    # validation for frequency


    if isinstance(frequency, (bool, np.bool_)):
        raise TypeError("frequency must be a number")
    
    if not isinstance(frequency, (int, float, np.integer, np.floating)):
        raise TypeError("frequency must be a number")

    if not np.isfinite(frequency):
        raise ValueError("frequency must be finite")
    
    if frequency < 0:
        raise ValueError("frequency must be non-negative")
   
    # validation for fs 
    
    if isinstance(fs, (bool, np.bool_)):
        raise TypeError("fs must be a number")
    
    if not isinstance(fs, (int, float, np.integer, np.floating)):
        raise TypeError("fs must be a number")

    if not np.isfinite(fs):
            raise ValueError("fs must be finite")
    
    if fs<=0:
        raise ValueError("fs must be greater than 0")
    
    
    # validation for n_samples

    if isinstance(n_samples, (bool, np.bool_)):
        raise TypeError("n_samples must be an integer")

    if not isinstance(n_samples, (int, np.integer, float, np.floating)):
        raise TypeError("n_samples must be an integer")

    if not np.isfinite(n_samples):
        raise ValueError("n_samples must be finite")

    if isinstance(n_samples, (float, np.floating)):
            if not n_samples.is_integer():
                n_samples = round(n_samples)
                warnings.warn(
                    "Fractional n_samples rounded to the nearest integer.", UserWarning
                )
            else:
                n_samples = int(n_samples)

    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")

    # validation for amplitude

    if isinstance(amplitude, (bool, np.bool_)):
        raise TypeError("amplitude must be a number")
    
    if not isinstance(amplitude, (int, float, np.integer, np.floating)):
        raise TypeError("amplitude must be a number")
    
    if not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite")

    # validation for phase

    if isinstance(phase, (bool, np.bool_)):
        raise TypeError("phase must be a number")
    
    if not isinstance(phase, (int, float, np.integer, np.floating)):
        raise TypeError("phase must be a number")
    
    if not np.isfinite(phase):
        raise ValueError("phase must be finite")

    # validation for offset

    if isinstance(offset, (bool, np.bool_)):
        raise TypeError("offset must be a number")
    
    if not isinstance(offset, (int, float, np.integer, np.floating)):
        raise TypeError("offset must be a number")
    
    if not np.isfinite(offset):
        raise ValueError("offset must be finite")

    # mathematical expression for sine wave

    x = amplitude*np.sin(2*np.pi*frequency*np.arange(n_samples)/fs + phase) + offset

    # create a Signal object with the generated sine wave data and the specified sampling frequency

    signal = Signal(x, fs)

    # check if frequency exceeds Nyquist frequency (fs/2) and issue a warning if it does

    if frequency > signal.fs/2:
            warnings.warn(
                "frequency exceeds Nyquist frequency (fs/2). This may cause aliasing.", UserWarning
    )
    
    return signal

# cosine function

def cosine(frequency, fs, n_samples, amplitude=1.0, phase=0.0, offset=0.0):

    # validation for frequency


    if isinstance(frequency, (bool, np.bool_)):
        raise TypeError("frequency must be a number")
    
    if not isinstance(frequency, (int, float, np.integer, np.floating)):
        raise TypeError("frequency must be a number")

    if not np.isfinite(frequency):
        raise ValueError("frequency must be finite")
       
    if frequency < 0:
        raise ValueError("frequency must be non-negative")
    
    # validation for fs 
    
    if isinstance(fs, (bool, np.bool_)):
        raise TypeError("fs must be a number")
    
    if not isinstance(fs, (int, float, np.integer, np.floating)):
        raise TypeError("fs must be a number")

    if not np.isfinite(fs):
            raise ValueError("fs must be finite")
    
    if fs<=0:
        raise ValueError("fs must be greater than 0")
    
    
    # validation for n_samples

    if isinstance(n_samples, (bool, np.bool_)):
        raise TypeError("n_samples must be an integer")

    if not isinstance(n_samples, (int, np.integer, float, np.floating)):
        raise TypeError("n_samples must be an integer")

    if not np.isfinite(n_samples):
        raise ValueError("n_samples must be finite")

    if isinstance(n_samples, (float, np.floating)):
            if not n_samples.is_integer():
                n_samples = round(n_samples)
                warnings.warn(
                    "Fractional n_samples rounded to the nearest integer.", UserWarning
                )
            else:
                n_samples = int(n_samples)

    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")

    # validation for amplitude

    if isinstance(amplitude, (bool, np.bool_)):
        raise TypeError("amplitude must be a number")
    
    if not isinstance(amplitude, (int, float, np.integer, np.floating)):
        raise TypeError("amplitude must be a number")
    
    if not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite")

    # validation for phase

    if isinstance(phase, (bool, np.bool_)):
        raise TypeError("phase must be a number")
    
    if not isinstance(phase, (int, float, np.integer, np.floating)):
        raise TypeError("phase must be a number")
    
    if not np.isfinite(phase):
        raise ValueError("phase must be finite")

    # validation for offset

    if isinstance(offset, (bool, np.bool_)):
        raise TypeError("offset must be a number")
    
    if not isinstance(offset, (int, float, np.integer, np.floating)):
        raise TypeError("offset must be a number")
    
    if not np.isfinite(offset):
        raise ValueError("offset must be finite")

    # mathematical expression for cosine wave

    x = amplitude*np.cos(2*np.pi*frequency*np.arange(n_samples)/fs + phase) + offset

    # create a Signal object with the generated cosine wave data and the specified sampling frequency

    signal = Signal(x, fs)

    # check if frequency exceeds Nyquist frequency (fs/2) and issue a warning if it does

    if frequency > signal.fs/2:
            warnings.warn(
                "frequency exceeds Nyquist frequency (fs/2). This may cause aliasing.", UserWarning
    )
    
    return signal


