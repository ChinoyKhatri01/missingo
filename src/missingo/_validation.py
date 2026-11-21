import numpy as np
import warnings

_NotProvided = object()

def validate_frequency(frequency):
    if isinstance(frequency, (bool, np.bool_)):
        raise TypeError("frequency must be a number")
        
    if not isinstance(frequency, (int, float, np.integer, np.floating)):
        raise TypeError("frequency must be a number")

    if not np.isfinite(frequency):
        raise ValueError("frequency must be finite")
        
    if frequency < 0:
        raise ValueError("frequency must be non-negative")

def validate_fs(fs):
    if isinstance(fs, (bool, np.bool_)):
        raise TypeError("fs must be a number")
         
    if not isinstance(fs, (int, float, np.integer, np.floating)):
        raise TypeError("fs must be a number")
     
    if not np.isfinite(fs):
        raise ValueError("fs must be finite")
         
    if fs<=0:
        raise ValueError("fs must be greater than 0")

def validate_n_samples(n_samples):
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

    return int(n_samples)

def validate_amplitude(amplitude):
    if isinstance(amplitude, (bool, np.bool_)):
        raise TypeError("amplitude must be a number")
        
    if not isinstance(amplitude, (int, float, np.integer, np.floating)):
        raise TypeError("amplitude must be a number")
        
    if not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite")

def validate_phase(phase):
    if isinstance(phase, (bool, np.bool_)):
        raise TypeError("phase must be a number")
        
    if not isinstance(phase, (int, float, np.integer, np.floating)):
        raise TypeError("phase must be a number")
        
    if not np.isfinite(phase):
        raise ValueError("phase must be finite")

def validate_offset(offset):
    if isinstance(offset, (bool, np.bool_)):
        raise TypeError("offset must be a number")
        
    if not isinstance(offset, (int, float, np.integer, np.floating)):
        raise TypeError("offset must be a number")
        
    if not np.isfinite(offset):
        raise ValueError("offset must be finite")

def nyquist_frequency_check(frequency, fs):
    if frequency > fs/2:
        warnings.warn(
            "frequency exceeds Nyquist frequency (fs/2). This may cause aliasing.", UserWarning
        )
def validate_inputs_periodic(frequency, fs, n_samples, amplitude, phase, offset):
    validate_frequency(frequency)
    validate_fs(fs)
    nyquist_frequency_check(frequency, fs)
    n_samples = validate_n_samples(n_samples)
    validate_amplitude(amplitude)
    validate_phase(phase)
    validate_offset(offset)

    return n_samples

def validate_duty(duty):
    if isinstance(duty, (bool, np.bool_)):
        raise TypeError("duty must be a number")

    if not isinstance(duty, (int, float, np.integer, np.floating)):
        raise TypeError("duty must be a number")

    if not np.isfinite(duty):
        raise ValueError("duty must be finite")
    
    if duty <=0 or duty >= 1:
        raise ValueError("duty must be between 0 and 1")
def validate_slope(slope):
    if isinstance(slope, (bool, np.bool_)):
        raise TypeError("slope must be a number")
        
    if not isinstance(slope, (int, float, np.integer, np.floating)):
        raise TypeError("slope must be a number")
        
    if not np.isfinite(slope):
        raise ValueError("slope must be finite")

def validate_std(std):
    if isinstance(std, (bool, np.bool_)):
        raise TypeError("std must be a number")
            
    if not isinstance(std, (int, float, np.integer, np.floating)):
        raise TypeError("std must be a number")
            
    if not np.isfinite(std):
        raise ValueError("std must be finite")

    if std <=0 :
        raise ValueError("std must be greater than zero")

def validate_mean(mean):
    if isinstance(mean, (bool, np.bool_)):
        raise TypeError("mean must be a number")
            
    if not isinstance(mean, (int, float, np.integer, np.floating)):
        raise TypeError("mean must be a number")
            
    if not np.isfinite(mean):
        raise ValueError("mean must be finite")

def validate_seed(seed):
    if seed is None:
        return

    if isinstance(seed, (bool, np.bool_)):
        raise TypeError("seed must be a non-negative integer or None")

    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be a non-negative integer or None")

    if seed < 0:
        raise ValueError("seed must be non-negative")

    

def validate_inputs_aperiodic(fs, n_samples, amplitude=_NotProvided, offset= _NotProvided, slope=_NotProvided, std = _NotProvided, mean = _NotProvided, seed = _NotProvided):
    
    validate_fs(fs)
    
    n_samples = validate_n_samples(n_samples)
    
    if amplitude is not _NotProvided:
        validate_amplitude(amplitude)
    
    if offset is not _NotProvided:
        validate_offset(offset)
    
    if slope is not _NotProvided:
        validate_slope(slope)
    
    if std is not _NotProvided:
        validate_std(std)

    if mean is not _NotProvided:
        validate_mean(mean)

    if seed is not _NotProvided:
        validate_seed(seed)

    return n_samples

# def validate_delay(delay, n_samples):

#     if isinstance(delay, (bool, np.bool_)):
#         raise TypeError("delay must be an integer")

#     if not isinstance(delay, (int, np.integer)):
#         if delay.is_integer():
#             delay = int(delay)
#         else:
        
#             raise TypeError("delay must be an integer")
    
#     if delay < 0 or delay >= n_samples:
#         raise ValueError("delay must be a non-negative integer less than n_samples")


def validate_low(low):
    if isinstance(low, (bool, np.bool_)):
        raise TypeError("low must be a number")

    if not isinstance(low, (int, float, np.integer, np.floating)):
        raise TypeError("low must be a number")

    if not np.isfinite(low):
        raise ValueError("low must be finite")


def validate_high(high):
    if isinstance(high, (bool, np.bool_)):
        raise TypeError("high must be a number")

    if not isinstance(high, (int, float, np.integer, np.floating)):
        raise TypeError("high must be a number")

    if not np.isfinite(high):
        raise ValueError("high must be finite")


def validate_uniform_bounds(low, high):
    if high <= low:
        raise ValueError("high must be greater than low")


def validate_delay(delay, n_samples):

    if isinstance(delay, (bool, np.bool_)):
        raise TypeError("delay must be an integer")

    if isinstance(delay, (float, np.floating)):
        if not np.isfinite(delay):
            raise ValueError("delay must be finite")

        if not delay.is_integer():
            raise TypeError("delay must be an integer")

        delay = int(delay)

    elif not isinstance(delay, (int, np.integer)):
        raise TypeError("delay must be an integer")

    if delay < 0 or delay >= n_samples:
        raise ValueError(
            "delay must be a non-negative integer less than n_samples"
        )

    return int(delay)


    
    