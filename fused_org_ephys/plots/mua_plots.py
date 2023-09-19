"""
spike_plots.py

General modules for rasterplots.
"""


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.pyplot import style
from matplotlib.transforms import Bbox

from ..utils import (save_fig)

import numpy as np


###############################################################################
###############################################################################


def opto_rasterplot(chan_groups, mua_spikes, mua_network, baseline_optostim_times=None, drug_optostim_times=None,
                    xlim=[0, None], networkevent_ylim=[0, None], srate=30_000,
                    fname=None, save_folder=None, file_extension='.pdf', scalebar=True):
    """
    Generate a raster plot with optogenetic stimulating times.

    Arguments:
    ----------
    chan_groups : array
        Array of channel groups.
    mua_spikes : array
        Array of spike times.
    mua_network : array
        Array of network events.
    baseline_optostim_times : array
        Array of baseline optogenetic stimulation times.
    drug_optostim_times : array
        Array of drug optogenetic stimulation times.
    xlim : array
        X-axis limits.
    networkevent_ylim : array
        Y-axis limits for network events.
    srate : int
        Sampling rate.
    fname : str
        Filename to save figure as.
    save_folder : str
        Folder to save figure in.
    file_extension : str
        File extension to save figure as.
    scalebar : bool
        Whether to add a scalebar to the figure.
    """

    style.use('fused_org_ephys/plots/paper.mplstyle')
    fig, ax = plt.subplots(3, 1, figsize=(8, 6),  sharex=True, gridspec_kw={'height_ratios': [0.8, 4, 0.6]})

    # --------- raster plot
    # re-order channels by channel groups
    # (e.g. if channels are grouped by depth, then channels in the same depth will be plotted together)

    reordered_chans = _reorder_channels(chan_groups)

    # rasterplot
    ax[1] = rasterplot(mua_spikes[[reordered_chans]][0], ax=ax[1])
    ax[1].set_ylabel('Active channels')
    ax[1].set_xlim(xlim)

    # --------- optogenetic stimulation plot
    # add baseline opto stim tag

    if baseline_optostim_times is not None:
        ax[2].eventplot(baseline_optostim_times, color='blue')
        ax[2].set_yticks([])
        ax[2].set_yticklabels([])
        ax[2].spines['left'].set_visible(False)
        ax[2].spines['bottom'].set_visible(False)

    if drug_optostim_times is not None:
        ax[2].eventplot(drug_optostim_times, color='blue')
        ax[2].set_yticks([])
        ax[2].set_yticklabels([])
        ax[2].spines['left'].set_visible(False)
        ax[2].spines['bottom'].set_visible(False)

    # ---------- optogenetic stimulation patch
    xmin, xmax = ax[1].get_xlim()

    # baseline
    if baseline_optostim_times is not None:
        if (xmax >= baseline_optostim_times[-1]) and (xmin <= baseline_optostim_times[0]):
            opto_end = baseline_optostim_times[-1]
            opto_start = baseline_optostim_times[0]
        elif (xmax < baseline_optostim_times[-1]) and (xmin > baseline_optostim_times[0]):
            opto_start = xmin
            opto_end = xmax
        elif (xmax > baseline_optostim_times[-1]) and (xmin > baseline_optostim_times[0]):
            opto_start = xmin
            opto_end = baseline_optostim_times[-1]
        elif (xmax < baseline_optostim_times[-1]) and (xmin > baseline_optostim_times[0]):
            opto_start = baseline_optostim_times[0]
            opto_end = xmax
        else:
            opto_start = baseline_optostim_times[0]
            opto_end = xmax

        _opto_patch(fig, ax[0], opto_start, opto_end)

    # --------- network event vector
    time = np.arange(0, len(mua_network), 1)/srate
    ax[0].plot(time, mua_network, color='black', lw=1.2)

    ax[0].set_ylim(networkevent_ylim)
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[0].set_yticks([])
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['left'].set_visible(False)

    # --------- time scalebar
    if scalebar is True:
        fontprops = fm.FontProperties(size=16)
        scalebar_x = AnchoredSizeBar(ax[2].transData, 10, "10 s", 'lower left', frameon=False, sep=2,
                                     size_vertical=0.1, pad=-2, borderpad=0.2, label_top=False,
                                     fontproperties=fontprops)  # x-axis
        ax[2].add_artist(scalebar_x)

    # --------- save
    save_fig(fig, fname, save_folder=save_folder, file_extension=file_extension)


def rasterplot(spiketimes, ax=None):
    """
    Plot a raster plot of spike times.

    Arguments:
    ----------
    spiketimes : array
        Array of spike times.
    ax : matplotlib axis
        Axis to plot on.

    Returns:
    --------
    ax : matplotlib axis
        Axis with raster plot.

    Note:
    -----
    Rasterplots are generated through the MUASpikeManager
    class using active channel spike times only.
    """

    if ax is None:
        ax = plt.gca()  # if not given, get current axis

    # raster plot
    ax.eventplot(spiketimes, color='k', linelengths=0.5)

    # remove spines
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_xticklabels([])

    # channel ticks
    major_yticks = np.arange(0, len(spiketimes), 1)
    ax.set_yticks(major_yticks)

    # remove ticks and insert
    # numbers for each active channel
    # (e.g. if 4 channels are active, then the ticks will be 1, 2, 3, 4)
    ylabels = ['']*(len(spiketimes))
    ylabels[0] = 1  # +1 to account for python indexing
    ylabels[-1] = len(spiketimes)
    ax.set_yticklabels(ylabels)

    # labels
    ax.set_ylabel('Channels')

    return ax


def rasterplot_inlet(mua_spikes=None, network_events=None, axin_xlim=[647.2, 647.2+20],
                     srate=30_000, baseline_crop=[0, 100], optostim_times=None, baseline_opto_start=647.2,
                     save_folder=None, fname=None, file_extension='.pdf'):
    """
    Generate a raster plot with optogenetic stimulating times and a zoomed-in inlet.

    Arguments:
    ----------
    mua_spikes : array
        Array of spike times.
    network_events : array
        Array of network events.
    axin_xlim : array
        X-axis limits for inlet.
    srate : int
        Sampling rate.
    baseline_crop : array
        X-axis limits for cropped raster plot.
    optostim_times : array
        Array of optogenetic stimulation times.
    baseline_opto_start : float
        Baseline optogenetic stimulation start time.
    save_folder : str
        Folder to save figure in.
    fname : str
        Filename to save figure as.
    file_extension : str
        File extension to save figure as.

    Note:
    -----
    This function is used to generate the raster plot inset in Figure 2 of the manuscript.
    A 100 second window is zoomed in on to highlight the network events and active channels.
    A box is drawn within the 100 second window to highlight the corresponding raster plot inlet.
    """

    style.use('fused_org_ephys/plots/paper.mplstyle')
    fig, ax = plt.subplots(3, 2, figsize=(11, 6), sharex=False,
                           gridspec_kw={'width_ratios': [1, 0.3], 'height_ratios': [0.5, 4, 0.4]})

    # --------- raster plot
    # network events
    time = np.arange(0, len(network_events), 1)/srate
    ax[0, 0].plot(time, network_events, color='black', lw=1.2)
    ax[0, 0].set_ylim([-0.1, 10])
    ax[0, 0].set_xlim(baseline_crop)

    # raster plots using mua spikes from active channels
    rasterplot(mua_spikes, ax=ax[1, 0])
    ax[1, 0].set_ylabel('Active channels')
    ax[1, 0].set_xlim(baseline_crop)

    # attach optostim patch for 100 second window
    _opto_patch(fig, ax[1, 0], baseline_opto_start, baseline_opto_start+100, height=0.575, y=0.2)

    # boxes highlighting sections for corresponding rasterplots
    _crop_patch(fig, ax[1, 0], axin_xlim[0], axin_xlim[1])

    # attach optostim times
    ax[2, 0].eventplot(optostim_times, color='blue')
    ax[2, 0].set_xticks([])
    ax[2, 0].set_xticklabels([])
    ax[2, 0].set_yticks([])
    ax[2, 0].set_yticklabels([])
    ax[2, 0].spines['left'].set_visible(False)
    ax[2, 0].spines['bottom'].set_visible(False)
    ax[2, 0].set_xlim(baseline_crop)

    # aesthetics // remove ticks
    ax[0, 0].set_xticks([])
    ax[0, 0].set_xticklabels([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_yticklabels([])
    ax[0, 0].spines['left'].set_visible(False)
    ax[0, 0].spines['bottom'].set_visible(False)

    # --------- inlet
    # network events
    time = np.arange(0, len(network_events), 1)/srate
    ax[0, 1].plot(time, network_events, color='black', lw=1.2)
    ax[0, 1].set_ylim([-0.1, 10])
    ax[0, 1].set_xlim(axin_xlim)

    # raster plot
    rasterplot(mua_spikes, ax=ax[1, 1])
    ax[1, 1].set_ylabel('')
    ax[1, 1].set_xlim(axin_xlim)

    # attach optostim times
    ax[2, 1].eventplot(optostim_times, color='blue')
    ax[2, 1].set_xticks([])
    ax[2, 1].set_xticklabels([])
    ax[2, 1].set_yticks([])
    ax[2, 1].set_yticklabels([])
    ax[2, 1].spines['left'].set_visible(False)
    ax[2, 1].spines['bottom'].set_visible(False)
    ax[2, 1].set_xlim(baseline_crop)
    ax[2, 1].set_xlim(axin_xlim)

    # aesthetics // remove ticks
    ax[0, 1].set_xticks([])
    ax[0, 1].set_xticklabels([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_yticklabels([])
    ax[0, 1].spines['left'].set_visible(False)
    ax[0, 1].spines['bottom'].set_visible(False)

    # ------- scale bars
    fontprops = fm.FontProperties(size=16)

    # main rasterplot
    scalebar_x = AnchoredSizeBar(ax[0, 0].transData, 20, "20 sec.",
                                 loc='upper right',
                                 frameon=False, sep=1,
                                 size_vertical=0.1,
                                 pad=-0.8,
                                 label_top=True,
                                 fontproperties=fontprops,
                                 bbox_to_anchor=Bbox.from_bounds(0, 0, 0.65, 0.87),
                                 bbox_transform=ax[0, 0].figure.transFigure)  # x-axis
    ax[0, 0].add_artist(scalebar_x)

    # inlet rasterplot
    scalebar_x = AnchoredSizeBar(ax[0, 1].transData, 5, "5 sec.",
                                 loc='upper right',
                                 frameon=False, sep=1,
                                 size_vertical=0.1,
                                 pad=-0.8,
                                 label_top=True,
                                 fontproperties=fontprops,
                                 bbox_to_anchor=Bbox.from_bounds(0, 0, 0.88, 0.87),
                                 bbox_transform=ax[0, 1].figure.transFigure)  # x-axis
    ax[0, 1].add_artist(scalebar_x)

    # ------- add text
    xmin, xmax = ax[0, 0].get_xlim()
    ax[0, 0].text(xmin, 5, 'Population response')

    xmin, xmax = ax[2, 0].get_xlim()
    ax[2, 0].text(xmin, 0, '10 sec. interval\n500 millisec. duration', color='blue')

    # --------- save
    save_fig(fig, fname, save_folder=save_folder, file_extension=file_extension)


def rasterplot_drugs(mua_spikes=None, network_events=None, drug_optostim_times=None,
                     srate=30_000, baseline_crop=[0, 100], optostim_times=None, baseline_opto_start=647.2,
                     drug_opto_start=1000, save_folder=None, fname=None, file_extension='.pdf'):
    """
    Generate a raster plot with optogenetic stimulating times pre- and post-drug application.

    Module is adapted from rasterplot_inlet().

    Arguments:
    ----------
    mua_spikes : array
        Array of spike times.
    network_events : array
        Array of network events.
    drug_optostim_times : array
        Array of drug optogenetic stimulation times.
    srate : int
        Sampling rate.
    baseline_crop : array
        X-axis limits for cropped raster plot.
    optostim_times : array
        Array of optogenetic stimulation times.
    baseline_opto_start : float
        Baseline optogenetic stimulation start time.
    drug_opto_start : float
        Drug optogenetic stimulation start time.
    save_folder : str
        Folder to save figure in.
    fname : str
        Filename to save figure as.
    file_extension : str
        File extension to save figure as.

    Note:
    -----
    This module is specific to DR022 and for other recordings may need to be adapted.
    """

    style.use('fused_org_ephys/plots/paper.mplstyle')
    fig, ax = plt.subplots(3, 3, figsize=(15, 5), sharey=False,
                           gridspec_kw={'width_ratios': [1, 0.1, 0.5],
                                        'height_ratios': [0.5, 4, 0.4]})

    # ------ baseline + optostim
    # rasterplot
    rasterplot(mua_spikes, ax=ax[1, 0])
    ax[1, 0].set_ylabel('Active channels')
    ax[1, 0].set_xlim(baseline_crop)

    # attach optostim patch for 100 second window
    _opto_patch(fig, ax[1, 0], baseline_opto_start, baseline_opto_start+100, height=0.575, y=0.2)

    # network events
    time = np.arange(0, len(network_events), 1)/srate
    ax[0, 0].plot(time, network_events, color='black', lw=1.2)
    ax[0, 0].set_xlim(baseline_crop)
    ax[0, 0].set_ylim([-0.1, 15])
    ax[0, 0].text((baseline_opto_start-100)+40, 15, 'Baseline', color='black')
    ax[0, 0].text(baseline_opto_start+35, 15, 'Stimulation', color='royalblue')

    # aesthetics // remove ticks
    ax[0, 0].set_xticks([])
    ax[0, 0].set_xticklabels([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_yticklabels([])
    ax[0, 0].spines['left'].set_visible(False)
    ax[0, 0].spines['bottom'].set_visible(False)

    # baseline opto stim times
    ax[2, 0].eventplot(optostim_times, color='blue')
    ax[2, 0].set_xticks([])
    ax[2, 0].set_xticklabels([])
    ax[2, 0].set_yticks([])
    ax[2, 0].set_yticklabels([])
    ax[2, 0].spines['left'].set_visible(False)
    ax[2, 0].spines['bottom'].set_visible(False)
    ax[2, 0].set_xlim([baseline_opto_start-100, baseline_opto_start+100])

    # ------ drugs + optostim
    rasterplot(mua_spikes, ax=ax[1, 2])
    ax[1, 2].set_xlim([drug_optostim_times[0]-0.5, drug_optostim_times[0]+99.5])
    ax[1, 2].set_ylabel('')

    # attach optostim patch for 100 second window
    _opto_patch(fig, ax[1, 2], drug_optostim_times[0]-0.5, drug_optostim_times[0]+99.5, height=0.575, y=0.2)

    # network events
    time = np.arange(0, len(network_events), 1)/srate
    ax[0,2].plot(time, network_events, color='black', lw=1.2)
    ax[0,2].set_xlim([drug_optostim_times[0]-0.5, drug_optostim_times[0]+99.5])  
    ax[0,2].set_ylim([-0.1, 15])
    ax[0,2].text(drug_opto_start-3, 15, 'AP5, CNQX, GZ, DR1 + DR2 inihibitors', color='tab:red')

    # aesthetics // remove ticks
    ax[0, 2].set_xticks([])
    ax[0, 2].set_xticklabels([])
    ax[0, 2].set_yticks([])
    ax[0, 2].set_yticklabels([])
    ax[0, 2].spines['left'].set_visible(False)
    ax[0, 2].spines['bottom'].set_visible(False)

    # drug opto stim times
    ax[2, 2].eventplot(drug_optostim_times, color='blue')
    ax[2, 2].set_xticks([])
    ax[2, 2].set_xticklabels([])
    ax[2, 2].set_yticks([])
    ax[2, 2].set_yticklabels([])
    ax[2, 2].spines['left'].set_visible(False)
    ax[2, 2].spines['bottom'].set_visible(False)
    ax[2, 2].set_xlim([drug_optostim_times[0]-0.5, drug_optostim_times[0]+99.5])

    # ---- blank insert to separate subplots

    # aesthetics // remove ticks
    for i in range(3):
        ax[i, 1].set_xticks([])
        ax[i, 1].set_xticklabels([])
        ax[i, 1].set_yticks([])
        ax[i, 1].set_yticklabels([])
        ax[i, 1].spines['left'].set_visible(False)
        ax[i, 1].spines['bottom'].set_visible(False)

    # seperating text between subplots
    ax[1, 1].text(0, 0.1, '+ Synaptic blocker cocktail', color='tab:red', rotation=270)

    # --------- scale bars
    fontprops = fm.FontProperties(size=16)

    scalebar_x = AnchoredSizeBar(ax[0, 0].transData, 20, "20 sec.",
                                 loc='upper left',
                                 frameon=False, sep=2,
                                 size_vertical=0.1,
                                 pad=-0.8,
                                 label_top=True,
                                 fontproperties=fontprops,)
    ax[0, 0].add_artist(scalebar_x)

    # ------------ text
    xmin, xmax = ax[2, 0].get_xlim()
    ax[2, 0].text(xmin, 0, '10 sec. interval\n500 millisec. duration', color='blue')

    # ------- save
    save_fig(fig, fname, file_extension=file_extension, save_folder=save_folder)


def _opto_patch(fig, ax, opto_start, opto_end, y=0.25, height=0.475):
    """
    Add optogenetic stimulation patch to raster plot.

    Arguments
    ----------
    fig : matplotlib figure
        Figure to add patch to.
    ax : matplotlib axis
        Axis to add patch to.
    opto_start : float
        Optogenetic stimulation start time.
    opto_end : float
        Optogenetic stimulation end time.
    y : float
        Y-axis position of patch.
    height : float
        Height of patch.

    Returns
    -------
    None
    """

    trans = matplotlib.transforms.blended_transform_factory(ax.transData, fig.transFigure)
    rect = mpatches.Rectangle(xy=(opto_start, y), width=(opto_end - opto_start),
                              height=height, transform=trans, ec='black', fc='orange', alpha=0.1,
                              lw=1, clip_on=False)
    fig.add_artist(rect)


def _reorder_channels(chan_groups):
    """
    Reorder channels by channel groups.

    Arguments
    ----------
    chan_groups : array
        Array of channel groups.

    Returns
    -------
    reordered : array
        Reordered array of channel groups.
    """

    reordered = []
    for chan_group in np.unique(chan_groups):
        chan_group_idx = np.where(chan_groups == chan_group)[0]
        reordered.append(chan_group_idx)

    reordered = np.concatenate(reordered)

    return reordered


def _crop_patch(fig, ax, start, end):
    """
    Crop raster plot to zoom in on a specific time period.

    Arguments:
    ----------
    fig : matplotlib figure
        Figure to add patch to.
    ax : matplotlib axis
        Axis to add patch to.
    start : float
        Start time.
    end : float
        End time.

    Returns:
    --------
    None
    """

    trans = matplotlib.transforms.blended_transform_factory(ax.transData, fig.transFigure)
    rect = mpatches.Rectangle(xy=(start, 0.2), width=(end - start),
                              height=0.575, transform=trans, ec='silver',
                              fill=None, lw=2, ls='--', clip_on=False)
    fig.add_artist(rect)
