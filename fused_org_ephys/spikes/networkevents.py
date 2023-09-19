"""
networkevents.py

Modules for processing network event
vectors from multi-unit activity spikes.
"""

import numpy as np
import warnings

# DeprecationWarning: elementwise comparison failed; this will raise an error in the future
warnings.filterwarnings("ignore", category=DeprecationWarning)


###################################################################################################
###################################################################################################


def return_mua_network(mua_spikes=None, srate=30_000):
    """
    Returns network event vector from multi-unit activity spikes.

    Arguments
    ---------
    mua_spikes : list
        List of spike times for each channel.
    srate : int
        Sampling rate (in Hz) of recording.

    Returns
    -------
    network_events : np.array
        Array of network events.
    """

    # get max length of max spike times
    length = max(list(map(max, mua_spikes)))

    # number of active channels (rows)
    number_of_channels = len(mua_spikes)

    # create empty array
    network_events = np.zeros((number_of_channels, int(srate * length)), dtype=int)

    # binarize timestamps
    # add to empty network_event arr
    for chan in range(len(mua_spikes)):
        spike_timestamps = mua_spikes[chan]*srate

        for spike_loc in spike_timestamps:
            try:
                network_events[chan, int(spike_loc)] = 1  # insert 1 at spike loc for binary (0 or 1) events
            except IndexError:
                pass  # remove at end frames

    network_events_sum = network_events.sum(axis=0)  # sum each frame @ sampling rate resolution

    # sum across 1 ms sliding window
    # --------------------------------
    # 'same' was used to not remove edges
    # note edge effects :: https://stackoverflow.com/questions/13728392/moving-average-or-running-mean

    network_events = np.convolve(network_events_sum, np.ones(int(0.001 * srate), dtype=int), mode='same')

    # smoothed with mean sliding window
    # ----------------------------------
    # smooth across N using numpy convolve
    # note each window is averaged

    N = int(0.1*srate)  # 100 ms
    network_events = np.convolve(network_events, np.ones(N)/N, mode='same') # normalized kernel

    return network_events
