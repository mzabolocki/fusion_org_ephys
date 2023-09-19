"""
ca_plots.py

Modules for calcium imaging plots.
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import style
from scipy import signal

from ..utils import (save_fig)


###############################################################################
###############################################################################


def plot_catrace(ca_traces, ca_peaks, neuron=0, show_peaks=True, save_folder=None,
                 fname=None, file_extension='.pdf'):
    """
    Plot calcium trace for a given neuron.

    Arguments:
    ----------
    ca_traces : array
        Array of calcium traces.
    ca_peaks : array
        Array of calcium peaks.
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

    style.use('fused_org_ephys/plots/paper.mplstyle')
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))

    # plot calcium trace
    sweep_vals = ca_traces[neuron]
    ax.plot(sweep_vals, c='royalblue', lw=1)

    # title
    ax.set_title(f'Neuron {neuron}')

    # labels
    ax.set_ylabel('Normalized fluorescence (a.u.)')
    ax.set_xlabel('Frames')

    # show peaks and widths
    if show_peaks is True:

        # plot peaks
        ax.scatter(ca_peaks[neuron], sweep_vals[ca_peaks[neuron]], c='r')

        # plot peak widths
        width = signal.peak_widths(sweep_vals, ca_peaks[neuron], rel_height=0.99)
        ax.hlines(*width[1:], color="orange", linewidth=2)

    # ------ save
    save_fig(fig, fname, save_folder=save_folder, file_extension=file_extension)
