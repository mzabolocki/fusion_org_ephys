"""
mua_thresh.py

Modules for preprocessing tools (e.g. thresholds for mua
spike detections)

"""

import numpy as np

################################################################
################################################################


def find_mua_thresh(voltage_set=None, thresh_detect_mode='donoho', upthr=5):
    """
    Find thresholds for mua detections.

    Parameters
    ----------
    voltage_set : list
        List of voltages to use for mua detections.
    **kwargs : dict
        Keyword arguments for mua_thresh.

    Returns
    -------
    mua_thresh : dict
        Dictionary of thresholds for mua detections.
    """

    # calculate std using donoho's rule
    # mutiply resulting std by upper and lower threshold limits

    if thresh_detect_mode == 'donoho':
        try:

            mua_thr_high = np.median(abs(voltage_set), axis=0)/0.6745
            mua_thr_high = (mua_thr_high*upthr)

            return mua_thr_high

        except TypeError:
            raise TypeError("channel not found | check 'good chans'")
    else:
        raise TypeError("'thresh_detect_mode' must be 'donoho'")


def rms(data, segment):
    """
    Calculates the square root of the mean square (RMS)
    along the segment of data.

    Arguments
    ---------
    data : arr
        Array of data.
    segment : int
        Segment of data to calculate RMS.

    Returns
    -------
    RMS of data.

    """

    a2 = np.power(data, 2)
    kernel = np.ones(segment)/float(segment)

    return np.sqrt(np.convolve(a2, kernel, 'same'))
