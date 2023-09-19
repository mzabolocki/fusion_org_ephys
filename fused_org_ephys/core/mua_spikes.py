"""
mua_spikes.py

Class for mua spike loading and analysis
from Cambridge Neurotech SiProbe recordings.
"""

import pandas as pd
import numpy as np
import pickle as pkl

from ..core.loader import (LoadMUASpikes)
from ..curation import (filter_activechannels)
from ..opto import (optostim_times)
from ..spikes import (return_responding_channels, return_mua_network, return_mfr)
from ..utils import (save_df, save_pickle)
from ..plots import (opto_rasterplot)


###################################################################################################
###################################################################################################


class MUASpikeManager(LoadMUASpikes):
    """
    Class for loading and processing mua spikes.

    Arguments:
    ----------
    metadata_path : str
        path to metadata file.
    base_folder : str
        base folder for data.
    opto_time_file_suffix : str
        suffix for optogenetic time file.
    opto_text_file_suffix : str
        suffix for optogenetic text file.
    time_range : list
        time range (in ms) for data.
    srate : int
        sampling rate (in Hz) for data.
    common_reference : str
        common reference for data.
    upthr : int
        threshold for spike detection.
    thresh_detect_mode : str
        threshold detection mode.
    save_folder : str
        folder to save processed data.
    spikearr_fpath : str
        file path to spike array.
    expID : str
        experiment ID.

    Notes:
    ------
    OpenEPhys GUI saves timestamps and txt files from aquisition
    in .../Message_Center-904.0/TEXT_group_1 folders. These are
    used to extract optogenetic stimulation times and/or drug application times.

    Sync processor delays sometimes found with the Open Ephys GUI
    must be accounted for and adjusted in the *.npy files prior to initiating the class.
    """

    def __init__(self, metadata_path=False, base_folder='Daniel_SiPro_Data/Data',
                 opto_time_file_suffix='timestamps_update', opto_text_file_suffix='text_update',
                 time_range=[0, None], srate=30_000, common_reference='median',
                 upthr=5, thresh_detect_mode='donoho', save_folder='/Processed/revision2/muaspikes/',
                 spikearr_fpath=None, expID=None):
        """ Initialize class with parameters."""

        super().__init__(metadata_path, base_folder, opto_time_file_suffix,
                         opto_text_file_suffix, time_range, srate,
                         common_reference, upthr, thresh_detect_mode, save_folder)

        ####################
        # LOAD SPIKE ARRAY #
        ####################
        # NOTE::
        # load spike array exported from MUASpikeManager class from spikearr_fpath

        if (spikearr_fpath is not None) and (expID is None):

            # load mua spike array
            self.spikearr_fpath = spikearr_fpath
            print(f'loading {self.spikearr_fpath}')
            with open(self.spikearr_fpath, 'rb') as file:
                self.mua_data = pkl.load(file)

            # metadata
            self.srate = srate
            self.metadata = self.mua_data[0]

            # mua spike arr data for single expID
            # from loaded mua spike array file
            self.mua_spikes = self.mua_data[1]
            self.bpass = self.mua_data[2]
            self.chans = self.mua_data[3]
            self.chan_groups = self.mua_data[4]
            self.baseline_threshs = self.mua_data[5]

        #######################
        # EXTRACT SPIKE ARRAY #
        #######################
        # NOTE ::
        # this is conducted for a single file path to binary files
        # and not load preprocessed spike array from a spikearr_fpath

        elif (spikearr_fpath is None) and (expID is not None):

            # extract mua spike data for single expID
            self.mua_data = self.get_mua_spikearr(expID=expID)

            # metadata
            self.srate = srate
            self.metadata = self.mua_data[0]

            # mua spike data
            self.mua_spikes = self.mua_data[1]
            self.bpass = self.mua_data[2]
            self.chans = self.mua_data[3]
            self.chan_groups = self.mua_data[4]
            self.baseline_threshs = self.mua_data[5]

    def set_activechannels(self, spk_activechan_interval=[None, None], spk_count_thresh=50):
        """
        Set active channels for mfr calculations.

        Arguments:
        ----------
        spk_activechan_interval : list
         spike active channel interval (in ms).
        spk_count_thresh : int
         threshold for number of spikes to be included on the active channel.
         Default to 50 spks (e.g. if interval is 600 seconds, this is 5 spks/minute).
        """

        # parameters
        self.spk_activechan_interval = spk_activechan_interval
        self.spk_count_thresh = spk_count_thresh

        # find active channel indices
        self.active_chans_idx = filter_activechannels(self.mua_spikes,
                                                      timeinterval=self.spk_activechan_interval,
                                                      spk_count_thresh=self.spk_count_thresh)

        # adjust mua spike arrays for active channel indices
        self.active_chans_mua_spikes = np.array(self.mua_spikes, dtype=object)[self.active_chans_idx]

    @property
    def set_network_event(self):
        """
        Generate and set network event array
        using active channels mua spikes.
        """

        try:
            self.active_chans_mua_spikes
        except AttributeError:
            print('set active channels')

        print('generating network event vector using active channels')

        self.network_events = return_mua_network(mua_spikes=self.active_chans_mua_spikes,
                                                 srate=self.srate)

    def set_baseline_optostim(self, start=None, train_dur=None, interval=None, dur=None):
        """
        Set baseline optostim pulses.

        Arguments:
        ----------
        start : int
         start time (in ms) for baseline optostim.
        train_dur : int
         duration (in ms) of baseline optostim.
        interval : int
         interval (in ms) between optostims.
        dur : int
         duration (in ms) of each optostim pulse.
        """

        # get optostim times at baseline
        self.baseline_optostim_times = optostim_times(start=start,
                                                      train_dur=train_dur,
                                                      interval=interval,
                                                      dur=dur,
                                                      srate=self.srate)

    def set_drug_optostim(self, start=None, train_dur=None, interval=None, dur=None):
        """
        Set drug optostim pulses.

        Arguments:
        ----------
        start : int
         start time (in ms) for drug optostim.
        train_dur : int
         duration (in ms) of drug optostim.
        interval : int
         interval (in ms) drug optostims.
        dur : int
         duration (in ms) of each optostim pulse.
        """

        # get optostim times at selected drug application
        self.drug_optostim_times = optostim_times(start=start,
                                                  train_dur=train_dur,
                                                  interval=interval,
                                                  dur=dur,
                                                  srate=self.srate)

    def save_muaspikes(self, fname=None):
        """
        Save mua spike array.

        Arguments:
        ----------
        fname : str
            filename for the mua spike array.

        Notes:
        ------
        The mua spike array is saved as a pickle file.
        No other file formats are supported.
        The extension is automatically added to the filename.
        """

        save_pickle(file=fname, base_folder=self.base_folder,
                    save_folder=self.save_folder,
                    data_arr=self.mua_data)

    def return_mfr(self, time_window=[None, None]):
        """
        Return mfr (mean firing rate) for active and
        all channels.

        Arguments:
        ----------
        time_window : int
            the time window (in ms) to calculate the mean firing rates (Hz)

        Returns:
        --------
        mfr_chans : np.array
         mean firing rate across all channels.

        mfr_activechans : np.array
         mean firing rate across active channels.

        """

        # mfr calc across active channels
        mfr_activechans = return_mfr(self.active_chans_mua_spikes, time_window)

        # mfr calc across all channels
        mfr_chans = return_mfr(self.mua_spikes, time_window)

        return mfr_chans, mfr_activechans

    def opto_rasterplot(self, xlim=[0, None], networkevent_ylim=[0, None],
                        scalebar=True, fname=None, save_folder=None, file_extension='.pdf'):
        """
        Plot opto raster with active channels.

        Arguments:
        ----------
        xlim : list
            x-axis limits for the raster plot.
        networkevent_ylim : list
            y-axis limits for the network event plot.
        scalebar : bool
            whether to plot scalebar.
        fname : str
            filename for the raster plot.
        save_folder : str
            folder to save the raster plot.
        file_extension : str
            file extension for the raster plot.

        Note:
        -----
        Rasterplots are generated using active channel spike times only.
        """

        try:
            self.active_chans_mua_spikes
            self.active_chans_idx
            self.baseline_optostim_times
            self.network_events
        except Exception as e:
            print(e)

        # drug optostim
        try:
            self.drug_optostim_times
        except AttributeError:
            self.drug_optostim_times = None

        print('plotting opto raster with active channels')

        # opto rasterplot
        opto_rasterplot(self.chan_groups[self.active_chans_idx], self.active_chans_mua_spikes,
                        self.network_events, self.baseline_optostim_times,
                        self.drug_optostim_times, xlim=xlim, networkevent_ylim=networkevent_ylim,
                        srate=self.srate, fname=fname,
                        save_folder=save_folder, file_extension=file_extension, scalebar=scalebar)

    def return_mfr_df(self, time_window=600, save_folder=None):
        """
        Return the mean firing rates (Hz) pre- and post-
        optogenetic stimulation within a given time window.

        For example, a time_window of 600 ms will calculate
        the mean firing rates (Hz) 600 ms before and after the first opto stimulation pulse.

        Descriptions of the dataframe columns are commented below.

        Arguments
        ---------
        time_window : int
            the time window (in ms) to calculate the mean firing rates (Hz)
        save_folder : str
            folder to save the dataframe

        Returns
        -------
        mfr_df : pd.DataFrame
            dataframe with mean firing rate (Hz) changes
            pre- and post-optogenetic stimulus.
        """

        try:
            self.active_chans_mua_spikes
            self.active_chans_idx
            self.baseline_optostim_times
        except Exception as e:
            print(e)

        df = pd.DataFrame(index=[self.metadata.file])

        # ---------- + org metadata + ----------
        # NOTE ::
        # expID --> experiment ID
        # exp_day --> experiment day
        # file --> file name
        # EB --> embryoid body
        # organoid --> organoid ID number
        # ages --> age of organoid
        # optostim_region --> optogenetic stimulation region (e.g. dorsal, ventral, cortical)
        # probe_region --> probe region (e.g. dorsal, ventral, cortical)

        for metadata_features in ['expID', 'exp_day', 'file', 'EB',
                                  'organoid', 'ages', 'optostim_region', 'probe_region']:
            df[metadata_features] = self.metadata[metadata_features].values[0]

        # ------ + channnel information + --------
        # NOTE ::
        # chans --> all channels (post processing)
        # number_chans --> number of channels (post processing)
        # active_chans --> active channel IDs (> 10 spks/min)
        # number_active_chans --> number of active channels (> 10 spks/min)

        df['chans'] = [self.chans]
        df['number_chans'] = len(self.chans)
        df['active_chans'] = [self.chans[self.active_chans_idx]]
        df['number_active_chans'] = len(self.active_chans_idx)

        # ------ + mfr time selection + --------
        # NOTE ::
        # mfr_start_time --> start time (in ms) for mfr calculation
        # mfr_end_time --> end time (in ms) for mfr calculation
        # mfr_opto_start_time --> start time (in ms) for mfr calculation post opto stim
        # mfr_opto_end_time --> end time (in ms) for mfr calculation post opto stim

        optostim_start_baseline = self.baseline_optostim_times[0]
        df['mfr_start_time'] = optostim_start_baseline-(time_window)
        df['mfr_end_time'] = optostim_start_baseline
        df['mfr_opto_start_time'] = optostim_start_baseline
        df['mfr_opto_end_time'] = optostim_start_baseline+time_window

        # ------ + mfr features + --------
        # NOTE ::
        # mfr --> mean firing rate (Hz) across all channels
        # mfr_opto --> mean firing rate (Hz) across all channels post opto stim
        # mfr_activechans --> mean firing rate (Hz) across active channels
        # mfr_opto_activechans --> mean firing rate (Hz) across active channels post opto stim
        # mfr_perchan --> mean firing rate (Hz) per channel
        # mfr_opto_perchan --> mean firing rate (Hz) per channel post opto stim
        # mfr_activechans_perchan --> mean firing rate (Hz) per active channel
        # mfr_opto_activechans_perchan --> mean firing rate (Hz) per active channel post opto stim

        # mfr calculation pre- and post- opto stim (active & non-active channels)
        mfr_chans_baseline_perchan, mfr_activechans_baseline_perchan = self.return_mfr([optostim_start_baseline-(time_window), optostim_start_baseline])
        mfr_chans_opto_perchan, mfr_activechans_opto_perchan = self.return_mfr([optostim_start_baseline, optostim_start_baseline+(time_window)])

        # mean firing rates (Hz) across all channels
        df['mfr'] = np.mean(mfr_chans_baseline_perchan)
        df['mfr_opto'] = np.mean(mfr_chans_opto_perchan)
        df['mfr_activechans'] = np.mean(mfr_activechans_baseline_perchan)
        df['mfr_opto_activechans'] = np.mean(mfr_activechans_opto_perchan)

        # mean firing rates (Hz) per chan
        df['mfr_perchan'] = [mfr_chans_baseline_perchan]
        df['mfr_opto_perchan'] = [mfr_chans_opto_perchan]
        df['mfr_activechans_perchan'] = [mfr_activechans_baseline_perchan]
        df['mfr_opto_activechans_perchan'] = [mfr_activechans_opto_perchan]

        # ----- + responding channels + -----
        # NOTE ::
        # responding_channels --> channels that respond to opto stim
        # responding_activechans --> active channels that respond to opto stim

        responding_channels = return_responding_channels(mfr_activechans_baseline_perchan, mfr_activechans_opto_perchan)
        df['responding_activechans'] = len(responding_channels)
        df['total_activechans'] = len(self.active_chans_idx)

        # ------ + save + --------
        # save the df to a set folder as an xlsx file with the file name
        # set to the file name of the recording

        if save_folder is not None:
            fname = 'mfr_df_' + str(self.metadata.file.values[0])
            save_df(df, fname, save_folder=save_folder)

        return df
