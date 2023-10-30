"""
active_channels.py

Modules to filter for active channels.
"""


###################################################################################################
###################################################################################################


def filter_activechannels(spiketimes, timeinterval=[0, None], spk_count_thresh=50):
    """
    Filter for active channels based on spike count within a time interval.

    Arguments
    ---------
    spiketimes : arr
        Array of spike times for each channel.
    timeinterval : list
        Time interval (in seconds) to filter for active channels.
    spk_count_thresh : int
        Spike count threshold to filter for active channels.

    Returns
    -------
    Updated spike times (arr) for active channels
    """

    # print out
    if timeinterval[1] is not None:
        print(f'filter channels with >= {(spk_count_thresh/(timeinterval[1] - timeinterval[0]))*60} spks/min | time interval {timeinterval} seconds')
    if timeinterval[1] is None:
        print(f'filter channels with >= {spk_count_thresh} spks across entire recording length')

    include_chan = []
    for chan in range(len(spiketimes)):

        count = 0
        for time in spiketimes[chan]:

            if timeinterval[1] is not None:
                if (time >= timeinterval[0]) & (time <= timeinterval[1]):
                    count += 1

            elif time >= timeinterval[0]:
                count += 1

        # NOTE::
        # filter chan bsaed on overal spike count
        # note that this is not scaled per/minute
        # please adjust spike count to account (e.g. >= 15 count_thresh over
        # a 60 second time interal is a spike rate >= 15/min)

        # filter channel
        if count >= spk_count_thresh:
            include_chan.append(chan)

    return include_chan
