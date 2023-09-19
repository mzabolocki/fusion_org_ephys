"""
smooth.py

Modules for signal smoothing.
"""

import numpy as np


###################################################################################################
###################################################################################################


def convolve(data=None, win_len=0.02, mode='valid', srate=30_000): 
    """
    Returns voltage array with np.convolve applied.
    (see numpy.convolve) for details.

    Arguments:
    ----------
    data : np.array
        Array of data.
    win_len : float
        Length of window (in seconds) to convolve.
    mode : str
        Mode of convolution (see numpy.convolve).
    srate : int
        Sampling rate (in Hz) of recording.

    Returns:
    --------
    data : np.array
        Array of data with np.convolve applied.

    References:
    -----------
    https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
    """

    if data is not None:

        data = np.convolve(data, np.ones(int(srate*win_len)), mode) / (int(srate*win_len)) # smoothing window + millisec conv. 

        return data
    else:
        raise ValueError(f'no data found for convolve')
