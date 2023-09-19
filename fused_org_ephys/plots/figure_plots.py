"""
figure_plots.py

General modules for figure plotting.

Plotting codes adapted from Jose Guzman.
"""

import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu as mwu
from scipy.stats import wilcoxon

from scipy.stats import sem
from scipy.stats import f_oneway

import numpy as np

###############################################################################
###############################################################################


def plot_pairs(xdata, ydata, labels, colors, ax=None):
    """
    Generate a bar plot from a list containing the data
    perform a Wilcoxon rank-test to test for paired differences.

    Arguments:
    ----------
    xdata   -- a list containing data to plot
    ydata   -- a list containing data to plot
    labels -- a list of string containig the variable names
    colors -- a list of strings containgin colors to plot the bars

    Returns:
    --------
    ax: a bar plot with the means, error bars with the standard error
    of the mean, and single data points.
    info: the mean and standard error of the samples, together with the
    the probability that the means are the same.
    """
    ax = ax or plt.gca()

    # single data points and error bars
    mycaps = dict(capsize=10, elinewidth=3, markeredgewidth=3)

    ax.plot(0, np.mean(xdata), '', color=colors[0])
    ax.errorbar(0, np.mean(xdata), sem(xdata), **mycaps, color=colors[0])

    ax.plot(1, np.mean(ydata), '', color=colors[1])
    ax.errorbar(1, np.mean(ydata), sem(ydata), **mycaps, color=colors[1])

    for i in zip(xdata, ydata):
        ax.plot([0, 1], i, color='gray', alpha=0.4, lw=0.5)

    # single data
    ax.plot(np.ones(len(xdata))*.0, xdata, 'o', color=colors[0], alpha=0.8)
    ax.plot(np.ones(len(ydata))*1, ydata, 'o', color=colors[1], alpha=0.8)

    # remove axis and adjust
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_yaxis().tick_left()

    # set axis ticks
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_xlim(-0.5, 1.5)

    # statistics
    stats_0 = (labels[0], np.mean(xdata), sem(xdata), len(xdata))
    stats_1 = (labels[1], np.mean(ydata), sem(ydata), len(ydata))
    print('%s = %2.4f +/- %2.4f, n = %d' %stats_0)
    print('%s = %2.4f +/- %2.4f, n = %d' %stats_1)
    w_test = wilcoxon(xdata, ydata, alternative='two-sided')[1]
    print(f'P = {w_test:2.4}, Wilcoxon signed-rank test (two-sided W test)\n')
    infostats = {'P-value': w_test}

    return ax, infostats


def plot_pairs_drugapp(wdata, xdata, ydata, labels, colors, ax=None):
    """
    Generate a bar plot from a list containing the data
    Adapted to accommodate 4 data sets.

    Arguments:
    ----------
    xdata   -- a list containing data to plot
    ydata   -- a list containing data to plot
    zdata  -- a list containing data to plot
    labels -- a list of string containig the variable names
    colors -- a list of strings containgin colors to plot the bars

    Returns:
    --------
    ax: a bar plot with the means, error bars with the standard error
    of the mean, and single data points.
    info: the mean and standard error of the samples, together with the
    the probability that the means are the same.
    """
    ax = ax or plt.gca()

    # single data points and error bars
    mycaps = dict(capsize=10, elinewidth=3, markeredgewidth=3)

    ax.plot(0, np.mean(wdata), '', color=colors[0])
    ax.errorbar(0, np.mean(wdata), sem(wdata), **mycaps, color=colors[0])

    ax.plot(1, np.mean(xdata), '', color=colors[1])
    ax.errorbar(1, np.mean(xdata), sem(xdata), **mycaps, color=colors[1])

    ax.plot(2, np.mean(ydata), '', color=colors[2])
    ax.errorbar(2, np.mean(ydata), sem(ydata), **mycaps, color=colors[2])

    for i in zip(wdata, xdata):
        ax.plot([0, 1], i, color='gray', alpha=0.4, lw=0.5)

    for i in zip(xdata, ydata):
        ax.plot([1, 2], i, color='gray', alpha=0.4, lw=0.5)

    # single data
    ax.plot(np.ones(len(wdata))*.0, wdata, 'o', color=colors[0], alpha=0.8)
    ax.plot(np.ones(len(xdata))*1, xdata, 'o', color=colors[1], alpha=0.8)
    ax.plot(np.ones(len(ydata))*2, ydata, 'o', color=colors[2], alpha=0.8)

    # remove axis and adjust
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_yaxis().tick_left()

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_xlim(-0.5, 2.5)

    # statistics anova
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
    F, p = f_oneway(wdata, xdata, ydata)

    print(f'statistic: {F}, p-value : {p}')

    infostats = {'statistic': F, 'p-value': p}

    return ax, infostats


def plot_boxes(xdata, ydata, labels, colors, ax=None):
    """
    Generate a box plot from a list containing the data 
    perform a Mann-Whitney U-test to test for mean differences.

    Arguments:
    ----------
    xdata   -- a list containing data to plot
    ydata   -- a list containing data to plot
    labels -- a list of string containig the variables
    colors -- a list of strings containg colors

    Returns:
    ax: a box plots with where the horizontal line is the
    median, boxes the first and third quartiles, and 
    the whiskers the most extreme data points <1.5x
    the interquartile distance form the edges. It also
    show single data form the experiments.

    info: the mean and standard error of the samples, together with the
    the probability that the means are the same.
    """
    if ax is None:
        ax = plt.gca()  # if not given, get current axis

    # box plots (sym = '' do not mark outliners)
    data = [xdata, ydata]
    bp = ax.boxplot(data, widths=0.45, patch_artist=1, sym='')
    # add sample size to labels

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_alpha(0.1)
        patch.set_linewidth(1)

    for patch in bp['whiskers']:
        patch.set(color='black', lw=1, ls='-')

    for cap in bp['caps']:
        cap.set(color='black', lw=1)

    for patch, color in zip(bp['medians'], colors):
        patch.set_color(color)
        patch.set_linewidth(1.5)

    # plot data points
    mean = 1
    for points, color in zip(data, colors):
        xval = np.random.normal(loc=mean, scale=.045, size=len(points))
        mean += 1
        ax.plot(xval, points, 'o', markersize=4, markeredgewidth=0.8, markeredgecolor=color, markerfacecolor='lightgrey')

    # remove axis and adjust
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_yaxis().tick_left()

    ax.set_xticklabels(labels, fontsize=14)
    ax.set_xticks([1,2])
    ax.xaxis.set_ticks_position('none')

    # statistics
    stats_0 = (labels[0], np.mean(data[0]), sem(data[0]), len(data[0]))
    stats_1 = (labels[1], np.mean(data[1]), sem(data[1]), len(data[1]))
    print('%s = %2.4f +/- %2.4f, n = %d' %stats_0)
    print('%s = %2.4f +/- %2.4f, n = %d' %stats_1)
    u_test = mwu(data[0], data[1], alternative='two-sided')[1]
    print('P = %2.4f, Mann-Whitney (two-sided U test)\n'%u_test)

    infostats = {'P-value': u_test}

    return (ax, infostats)