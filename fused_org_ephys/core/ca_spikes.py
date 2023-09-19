"""
ca_spikes.py

Class for calcium imaging spike loading and analysis
using calcium traces from caiman.

Code by Daniel Reumann, edited by Michael Zabolocki. 
"""

import os
import pandas as pd
import numpy as np
import pickle as pkl
from scipy import signal

from ..utils import (save_df)
from ..plots import (plot_catrace)


###################################################################################################
###################################################################################################


class CaSpikeManager():
    """
    Class to load, preprocess and analyze calcium imaging traces
    produced by caiman.

    Arguments:
    ----------
    traces_fname : str
        File name of calcium traces.
    prominence : float
        Prominence for calcium spike detection.
    rel_height : float
        Relative height for calcium spike detection.
    fs : float
        Sampling frequency of calcium traces.

    Notes:
    ------
    The calcium traces are preprocessed by normalizing and subtracting the baseline.

    The calcium spikes are detected using the scipy.signal.find_peaks function.

    References:
    -----------
    CalmAn code: https://github.com/flatironinstitute/CaImAn
    CalmAn paper: https://elifesciences.org/articles/38173
    """

    def __init__(self, traces_fname=None, prominence=0.4, rel_height=0.99, fs=0.065):
        """ Initialize class with parameters."""

        #############################
        # LOAD CALCIUM TRACES ARRAY #
        #############################

        self.traces_fname = traces_fname
        self.fs = fs

        if (traces_fname is not None):
            with open(traces_fname, 'rb') as f:

                # ----- load pickled calcium traces
                self.ca_traces = pkl.load(f)

                # ----- preprocess calcium traces
                self.normalized_subtraced_ca_traces = self._preprocess_traces()

                # ----- get calcium spikes
                self.ca_widths, self.ca_peaks = self.get_caspikes(prominence=prominence, rel_height=rel_height)

        else:
            raise ValueError('traces_fname must be provided')

    def _preprocess_traces(self):
        """
        Function to preprocess calcium traces.
        Performs normalization and baseline subtraction.

        Returns:
        --------
        normalized_subtraced_ca_traces : array
            Array of normalized and baseline subtracted calcium traces.
        """

        # --- normalize traces
        # NOTE ::
        # this normalizes all calcium traces and writes it
        # into the previously generated normalized_all array with new dimensions (e.g. 1,5900)

        range_to_normalize = (0, 1)
        normalized_ca_traces = np.zeros(shape=(1, self.ca_traces.shape[1]))
        for i in self.ca_traces:
            x = self._normalize(i, range_to_normalize[0], range_to_normalize[1])
            x = np.array([x])

            normalized_ca_traces = np.append(normalized_ca_traces, x, axis=0)

        # remove the first entry
        normalized_ca_traces = np.delete(normalized_ca_traces, (0), axis=0)

        # --- baseline subtraction
        # NOTE ::
        # a subtraction was necessary because caiman did not completely remove the background
        # from the recordings. A 5% was found to work well.

        normalized_subtraced_ca_traces = np.zeros(shape=(1, self.ca_traces.shape[1]))
        for i in normalized_ca_traces:
            x = np.asarray([i])
            x = np.where(x < 0.05, 0, x)
            normalized_subtraced_ca_traces = np.append(normalized_subtraced_ca_traces, x, axis=0)

        # remove the first entry
        normalized_subtraced_ca_traces = np.delete(normalized_subtraced_ca_traces, (0), axis=0)

        return normalized_subtraced_ca_traces

    def _normalize(self, trace_vals, t_min=0, t_max=1):
        """
        Function to normalize calcium traces between t_min and t_max.

        Arguments:
        ----------
        trace_vals : array
            Array of calcium trace.
        t_min : float
            Minimum value for normalization.
        t_max : float
            Maximum value for normalization.

        Returns:
        --------
        norm_arr : array
            Array of normalized calcium traces.
        """

        norm_arr = []
        diff = t_max - t_min
        diff_arr = max(trace_vals) - min(trace_vals)

        for i in trace_vals:
            if diff_arr > 0:
                temp = (((i - min(trace_vals))*diff)/diff_arr) + t_min
                norm_arr.append(temp)
            else:  # dead trace
                norm_arr.append(np.nan)

        return norm_arr

    def get_caspikes(self, prominence=0.4, rel_height=0.99):
        """
        Function to get calcium spike timestamps and widths from calcium traces.

        Arguments:
        ----------
        prominence : float
            Prominence for calcium spike detection.
        rel_height : float
            Relative height for calcium spike detection.

        Returns:
        --------
        widths : array
            Array of calcium spike widths.
        peaks : array
            Array of calcium spike peaks.

        Note:
        -----
        Always check the prominence and rel_height for quantificaions.
        Too high or too low values might result in wrong values.
        Default values are set and worked well for the data.
        """

        widths = []
        peaks = []
        for trace in self.normalized_subtraced_ca_traces:
            sig_peaks, _ = signal.find_peaks(trace, prominence=prominence)
            half_peak_res = signal.peak_widths(trace, sig_peaks, rel_height=rel_height)
            widths.append(half_peak_res[0])
            peaks.append(sig_peaks)

        return widths, peaks

    def plot_catrace(self, neuron=0, show_peaks=True, save_folder=None, fname=None, file_extension='.pdf'):
        """
        Plot calcium trace for a given neuron.

        Arguments:
        ----------
        neuron : int
            Neuron to plot.
        show_peaks : bool
            If True, show peaks and widths on plot.
        save_folder : str
            Folder to save figure to.
        fname : str
            File name to save figure to.
        file_extension : str
            File extension to save figure to.

        Returns:
        --------
        fig : matplotlib figure
            Figure with calcium trace.
        """

        plot_catrace(self.normalized_subtraced_ca_traces, self.ca_peaks,
                     neuron=neuron, show_peaks=show_peaks, save_folder=save_folder,
                     fname=fname, file_extension=file_extension)

    def return_caspikewidth_df(self, save_folder=None):
        """
        Return dataframe of calcium spike widths.

        Arguments:
        ----------
        save_folder : str
            Folder to save dataframe to.

        Note:
        -----
        Values below 1sec are removed in postprocessing for figure generations.
        These appeared as artefacts based on visual inspections.
        """

        caspike_df = pd.DataFrame(self.ca_widths)
        caspike_df = caspike_df*self.fs  # adjust for fs

        # adjust index
        indexes = ['neuron']*(caspike_df.shape[0])
        indexes = ['{}_{}'.format(i, s) for s, i in enumerate(indexes, start=0)]
        caspike_df.index = indexes

        # adjust column names
        cols = ['spike']*(caspike_df.shape[1])
        cols = ['{}_{}'.format(i, s) for s, i in enumerate(cols, start=0)]
        caspike_df.columns = cols

        # save
        if save_folder is not None:
            fname = os.path.basename(self.traces_fname)
            fname = fname.replace('.pickle', '')

            save_df(caspike_df, str(fname), save_folder=save_folder)

        return caspike_df
