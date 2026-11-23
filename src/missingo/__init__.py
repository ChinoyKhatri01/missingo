from .signal import Signal

from .td_operations import (
    amplitude_scaling,
    concatenate,
    pad,
    normalisation,
    offset_adjustment,
    crop_by_indices,
    crop_by_time,
    shift_circular,
    shift_extended,
    shift_fixed,
    reverse,
    segmentation_by_count,
    segmentation_by_length,
)

from .generator import(
    sine, 
    cosine, 
    square, 
    sawtooth, 
    triangle,
    impulse, 
    step, 
    ramp, 
    dc, 
    gaussian_pulse, 
    white_noise_gaussian, 
    white_noise_uniform
)

from . import td_operations

from .td_operations import *

from . import generator

from .generator import *