"""
opto.py

Contains modules for opto stimulation tag generation and analysis using Pulse Pal.

For more information regarding stimulation setup see: https://sites.google.com/site/pulsepalwiki/parameter-guide
"""

import numpy as np


###################################################################################################
###################################################################################################


def optostim_times(start, train_dur, interval, dur, srate=30000):
    """
    Generate opto stimulation time tags based on PulsePal inputs.
    Currently set only for a single pulse delivery (not a burst pulse).

    Arguments
    ---------
    start : int
        start time (in ms) for opto stim.
    train_dur : int
        duration (in ms) of opto stim.
    interval : int
        interval (in ms) between opto stims.
    dur : int
        duration (in ms) of each opto stim pulse.
    srate : int
        sampling rate (in Hz) of recording.

    Returns
    -------
    adjusted_optostim : np.array
        array of opto stim times (in seconds).
    """

    # adjust times
    end = start + train_dur  # find end of opto stim
    interval = interval + dur  # adjust interval time to + burst duration width

    # prelim
    optostim = np.arange(start, end, interval)

    # increase width of each stim
    if dur > 0:
        adjusted_optostim = []  # initialize
        for stim_start in optostim:

            stim_end = stim_start + dur
            adjusted_optostim.append(np.arange(stim_start, stim_end, 1/srate))

        adjusted_optostim = np.hstack(adjusted_optostim)

    else:
        adjusted_optostim = optostim  # no duration added

    return adjusted_optostim
