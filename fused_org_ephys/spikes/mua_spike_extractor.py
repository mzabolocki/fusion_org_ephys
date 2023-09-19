"""
mua_spike_extractor.py

Modules for spike timestamp extractions.

"""

import numpy as np
from numpy import absolute, sqrt, mean
from scipy import signal


###################################################################################################
###################################################################################################


def extract_mua_spikes(voltage_set=None, chans=None, mua_thresh_set=None, srate=30_000):
    """
    Extract spikes from mua voltage traces.

    Parameters
    ----------
    voltage_set : ndarray
        mua voltage traces
    chans : ndarray
        mua channels
    mua_thresh_set : ndarray
        mua thresholds (per channel)
    srate : int
        sampling rate

    Returns
    -------
    spike_times : ndarray
        mua spike times
    spike_chans : ndarray
        mua spike channels

    """

    spike_times = []
    if voltage_set is not None:
        if chans is not None:
            for count, chan in enumerate(chans):

                voltage = voltage_set[:, count]  # isolate voltages

                if len(voltage) >= 1:

                    # calculate mua threshold
                    # for spike detection
                    if mua_thresh_set is not None:
                        selected_mua_thresh = mua_thresh_set[count]
                    else:
                        raise ValueError('parse baseline thresholds for mua detection')

                    # calculate noise floor
                    noisefloor = sqrt(mean(absolute(voltage)**2))

                    # detect spikes in the absolute voltages
                    refractory_period = (2/1000)  # 2 milliseconds
                    peaks, _ = signal.find_peaks(x=abs(voltage), height=(noisefloor+selected_mua_thresh),
                                                 distance=(refractory_period*srate))

                    # NOTE: for low firing channels,
                    # donoho is set near noise floor
                    # and skews detections, to overcome this
                    # spikes >= 40 uV from the noise floor were
                    # only included for absolute voltages

                    idx = abs(voltage[peaks]) >= (noisefloor + 40)
                    peaks = peaks[idx]

                    # spike times arr
                    if len(peaks) >= 1:
                        times = np.arange(len(voltage))/srate  # time vector
                        spike_times.append(times[peaks])  # output times (s)

                    else:
                        spike_times.append([])
                else:
                    spike_times.append([])
            return spike_times
        else:
            raise ValueError(f'chans can not be: {chans}')
    else:
        raise ValueError('no mua voltage array found')
