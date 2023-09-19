"""
spike_features.py

Basic modules for spike feature extractions.
"""

import numpy as np

##########################################################################
##########################################################################


def return_mfr(spiketimes, time_window):
    """
    Returns mean firing rate (mfr) for each channel.

    Arguments
    ---------
    spiketimes : list
        List of spike times for each channel.
    time_window : list
        Time window (in seconds) to calculate mfr.

    Returns
    -------
    mfr : np.array
        Array of mfr for each channel.
    """

    mfr = []
    for chan in range(len(spiketimes)):

        count = 0
        for time in spiketimes[chan]:

            if time_window[1] is not None:
                if (time >= time_window[0]) & (time <= time_window[1]):
                    count += 1
            else:
                if (time >= time_window[0]):
                    count += 1

        mfr.append(count/(time_window[1] - time_window[0]))

    mfr = np.array(mfr)

    return mfr


def return_responding_channels(pre_mfr, post_mfr):
    """
    Returns channels with >= 10% mfr change.

    Arguments
    ---------
    pre_mfr : np.array
        Array of pre-stimulus mfr for each channel.
    post_mfr : np.array
        Array of post-stimulus mfr for each channel.

    Returns
    -------
    responding_channels : list
        List of channels with > 10% mfr change.
    """

    # normalize mfr to pre
    post_mfr_norm = post_mfr / np.mean(pre_mfr)

    responding_channels = []
    for chan in range(len(post_mfr_norm)):

        # channels with > 10% mfr change
        if post_mfr_norm[chan] >= 1.1:
            responding_channels.append(chan)

    return responding_channels
