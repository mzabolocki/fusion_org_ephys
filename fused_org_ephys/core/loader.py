"""
loader.py

Class for loading continuous.dat files
and saving extracted extracellular signals
and/or multi-unit-activity spikes.

"""

import numpy as np

import pandas as pd

import probeinterface as pi
import spikeinterface as si
import spikeinterface.preprocessing as spre

from .metadata import Metadata

from ..utils import (_fileinfo, _isolate_metadata, _check_frames, _check_reclength,
                     loader_report_info, save_pickle, _fileinfo, _isolate_metadata)
from ..preprocessing import (find_mua_thresh)
from ..spikes import (extract_mua_spikes)


###################################################################################################
###################################################################################################


class LoadChannels(Metadata):
    """
    Load class. Core object for loading and preprocessing extracellular signals
    recorded from Cambridge Neurotech Silicon Probes.

    This class is designed to be used with the Cambridge Neurotech Silicon Probes.
    It takes care of loading and preprocessing extracellular signals, by removing
    bad/noisey channels, applying common references, and wiring devices.

    Wiring is set to the left and right probes based on the ASSY-156>RHD2164 intan headstage
    and is conducted across 64 or 128 channels in the case of group probe recordings.

    Arguments:
    ----------
    metadata_path (str, optional):
        path to tissue culture metadata file. Defaults to None.
    base_folder (str, optional):
        base folder for binary files. Defaults to 'TSC2_Ephys_Data'.
    opto_time_file_suffix (str):
        suffix for timestamps.npy. Defaults to 'timestamps_update'.
    opto_text_file_suffix (str):
        suffix for sync_messages.txt. Defaults to 'text_update'.
    recording_segment_selection (str, int):
        recording segment selection ('baseline', 1, 2, 3 ...). Defaults to 'baseline'.
    time_range (list, optional):
        time range selection. Defaults to [None, None].
    srate (int, float, optional):
        sampling rate. Defaults to 30_000.
    save_folder (str, optional):
        folder for saving. Defaults to '/Processed/MUA_spikes'.

    Notes:
    ------
    OpenEPhys GUI saves timestamps and txt files from aquisition
    in .../Message_Center-904.0/TEXT_group_1 folders. These are
    used to extract optogenetic stimulation times and/or drug application times.

    Sync processor delays sometimes found with the Open Ephys GUI
    must be accounted for and adjusted in the *.npy files prior to initiating the class.
    """

    def __init__(self, metadata_path=None, base_folder='Daniel_SiPro_Data/Data',
                 opto_time_file_suffix='timestamps_update',
                 opto_text_file_suffix='text_update',
                 time_range=[None, None], srate=30_000, common_reference='median',
                 save_folder='/Processed/Time_series'):
        """ initialize object with desired settings. """

        super().__init__(metadata_path, base_folder, opto_time_file_suffix,
                         opto_text_file_suffix)

        self.time_range = time_range
        self.srate = srate
        self.common_reference = common_reference
        self.save_folder = save_folder

    def read_binary(self, file_path=None, nchan=None):
        """
        Reads continuous.dat files and returns a spikeinterface RecordingExtractor object.

        Parameters
        ----------
        file_path : str
            Path to the continuous.dat file.
        nchan : int
            Number of channels in the continuous.dat file.

        Returns
        -------
        si_recording : RecordingExtractor
            RecordingExtractor object.
        """

        if file_path is None:
            file_path = self.file_path
        if nchan is None:
            nchan = self.nchan

        # create the RecordingExtractor object
        gain_to_uV = 0.19499999284744262695  # gain to uV
        dtype = "int16"

        si_recording = si.read_binary(file_path, num_chan=nchan, sampling_frequency=self.srate,
                                      dtype=dtype, gain_to_uV=gain_to_uV, offset_to_uV=0,
                                      time_axis=0, is_filtered=None)

        return si_recording

    def set_probe(self, si_recording, nchan=None, left_probe=None, right_probe=None):
        """
        Sets the probe for the RecordingExtractor object.

        Parameters
        ----------
        si_recording : RecordingExtractor
            RecordingExtractor object.
        nchan : int
            Number of channels in the continuous.dat file.
        left_probe : str
            Name of the left probe.
        right_probe : str
            Name of the right probe.

        Returns
        -------
        si_recording : RecordingExtractor
            RecordingExtractor object.

        Note:
        -----
        Wiring is set to the left and right probes.
        based on the ASSY-156>RHD2164 intan headstage.
        All probes are Cambridgeneurotech's commercial probes.
        """

        # public library of commercial probes
        # (https://gin.g-node.org/spikeinterface/probeinterface_library/)

        manufacturer = 'cambridgeneurotech'

        # probe is interfaced to an Intan headstage
        # with 64 channels (“RHD2164”)
        wiring = 'ASSY-156>RHD2164'

        if nchan > 67:  # if 2 probes

            probegroup = pi.ProbeGroup()  # probe group

            # -------- probe 0
            probe0 = pi.get_probe(manufacturer, left_probe)
            probe0.wiring_to_device(wiring)
            probegroup.add_probe(probe0)

            # -------- probe 1
            # contact ids are moved + 64 from probe 0
            # to avoid contact ID clashes
            probe1 = pi.get_probe(manufacturer, right_probe)
            probe1.set_contact_ids(contact_ids=np.arange(65, 129, 1).astype(str))
            probe1.set_device_channel_indices(channel_indices=(probe0.device_channel_indices+64))
            probe1.move([-20, -500])  # ~ move based on recording set-up
            probegroup.add_probe(probe1)

            recording_prb = si_recording.set_probegroup(probegroup, group_mode="by_shank")

            return recording_prb

        else:  # 1 probe

            probe = pi.get_probe(manufacturer, left_probe)
            probe.wiring_to_device(wiring)
            recording_prb = si_recording.set_probe(probe, group_mode="by_shank")

            return recording_prb

    def remove_bad_channels(self, si_recording):
        """
        Removes bad channels from the RecordingExtractor object.

        Parameters
        ----------
        si_recording : RecordingExtractor
            RecordingExtractor object.

        Returns
        -------
        si_recording : RecordingExtractor
            RecordingExtractor object.
        """

        return spre.remove_bad_channels(si_recording, bad_threshold=2)

    @property
    def _check_start_end_times(self):
        """
        Hidden module to checks start and end times for loader and
        signal extractions.

        If None in time_range, set to full recording length.
        """

        # -------- start & end times
        # default start and end times to full
        # recording lenghts if None set
        if self.time_range[0] is None:
            start = None
        else:
            start = int(self.time_range[0]*self.srate)

        if self.time_range[1] is None:
            end = None
        else:
            end = int(self.time_range[1]*self.srate)

        return start, end

    def return_loader(self, **loader_kwargs):
        """
        Returns a Loader object.

        Applies preprocessing to the continuous.dat file and returns a Loader object.
        Bandpass filtering is applied to the continuous.dat file.
        Common reference is applied to the continuous.dat file, if common_reference is not None.

        Parameters
        ----------
        loader_kwargs : dict
            Keyword arguments to pass to the Loader object.

        Returns
        -------
        loader : Loader
            Loader object.
        """

        # 1) read binary
        si_recording = self.read_binary(file_path=loader_kwargs['file_path'], nchan=loader_kwargs['nchan'])

        # 2) set probe
        si_recording = self.set_probe(si_recording, nchan=loader_kwargs['nchan'],
                                      left_probe=loader_kwargs['left_probe'],
                                      right_probe=loader_kwargs['right_probe'])

        # 3) remove bad channels
        si_recording = self.remove_bad_channels(si_recording)

        # 4) checks
        start_frame, end_frame = _check_frames(si_recording, start=loader_kwargs['start'], end=loader_kwargs['end'])
        start_frame, end_frame = _check_reclength(si_recording, loader_kwargs['file_path'], start=start_frame, end=end_frame)

        # 5) slice recording
        si_recording = si_recording.frame_slice(start_frame=start_frame, end_frame=end_frame)

        # 6) bandpass
        # used for mua spike extractions (IIR)
        # 300 - 3000 Hz, 4th order butterworth
        si_recording = spre.bandpass_filter(si_recording, filter_order=4, freq_min=300, freq_max=3000)

        # 7) common reference
        if self.common_reference is not None:
            si_recording = spre.common_reference(si_recording, reference='global',
                                                 operator=self.common_reference)
        else:
            print('no common reference applied')

        return si_recording


###################################################################################################
###################################################################################################


class LoadMUASpikes(LoadChannels):
    """
    MUASpikes class for multi-unit spike extractions.

    Arguments:
    ----------
    metadata_path (str, optional):
        path to tissue culture metadata file. Defaults to None.
    base_folder (str, optional):
        base folder for binary files. Defaults to 'TSC2_Ephys_Data'.
    opto_time_file_suffix (str):
        suffix for timestamps.npy. Defaults to 'timestamps_update'.
    opto_text_file_suffix (str):
        suffix for sync_messages.txt. Defaults to 'text_update'.
    recording_segment_selection (str, int):
        recording segment selection ('baseline', 1, 2, 3 ...). Defaults to 'baseline'.
    time_range (list, optional):
        time range selection in seconds. Defaults to [0, None].
        seconds are automatically converted to frames.
    srate (int, float, optional):
        sampling rate. Defaults to 30_000.
    common_reference (str, optional):
        common reference (e.g. mean, median). Defaults to 'median'.
    upthr (int, optional):
        upper threshold for mua spike detection. Defaults to 5.
    thresh_detect_mode (str, optional):
        std calculation type for threshold detection (rms or donoho method). Defaults to 'donoho'.
    save_folder (str, optional):
        folder for saving. Defaults to '/Processed/MUA_spikes'.
    """

    def __init__(self, metadata_path=None, base_folder='Daniel_SiPro_Data/Data',
                 opto_time_file_suffix='timestamps_update', opto_text_file_suffix='text_update',
                 time_range=[0, None], srate=30_000, common_reference='median',
                 upthr=5, thresh_detect_mode='donoho', save_folder='/Processed/revision2/muaspikes/'):
        """ initialize object with desired settings. """

        super().__init__(metadata_path, base_folder, opto_time_file_suffix,
                         opto_text_file_suffix, time_range, srate, common_reference)

        self.upthr = upthr
        self.thresh_detect_mode = thresh_detect_mode
        self.save_folder = save_folder
        self.srate = srate
        self.upthr = upthr
        self.thresh_detect_mode = thresh_detect_mode

    def get_mua_spikearr(self, isolated_metadata=None, expID=None, file_path=None, nchan=None,
                         left_probe=None, right_probe=None):
        """
        Process muaspike data from baseline and drug selection times.

        Arguments:
        ----------
        isolated_metadata : bool
            isolated metadata for experiment ID
        expID : str
            experiment ID. Defaults to None.
        file_path : str
            file path to continous.dat file. Defaults to None.
        nchan : int
            number of channels on select probe. Defaults to None.
        left_probe : str
            left probe. Defaults to None.
        right_probe : str
            right probe. Defaults to None.

        Returns:
        --------
        muaspike_data_arr : arr
            nested arr of muaspike data stored
            in the following order:

            - muaspike_data_arr[0]: metadata
            - muaspike_data_arr[1]: mua spikes
            - muaspike_data_arr[2]: bandpass signals
            - muaspike_data_arr[3]: channel IDs
            - muaspike_data_arr[4]: channel groups
            - muaspike_data_arr[5]: mua thresholds

        Note:
        -----
        Muaspike data is detected above a threshold
        set at baseline periods.

        Detection is by default calculated as a multiple of
        the std, calculated using the donoho method.
        The median method is used to calculate the threshold.

        Spikes at detected > refractory periods, set at 1 millisecond.
        """

        # ---------- metadata
        # if isolated_metadata is None, find metadata for select expID
        # and isolate probe details and file path information
        if isolated_metadata is None:

            # return metadata for select expID
            isolated_metadata = (pd.DataFrame(self.metadata)[pd.DataFrame(self.metadata).expID.isin(expID)])

            # find file path info for expID
            fileinfo = _fileinfo(isolated_metadata, self.file_path_info)
            file_path = fileinfo['file_paths'][0]

            # find probe info
            left_probe = isolated_metadata.left_probe.values[0]
            right_probe = isolated_metadata.right_probe.values[0]
            nchan = int(isolated_metadata.nchan.values[0])

        else:  # continue with isolated_metadata from batch processing
            pass

        # ---------- loader
        start, end = self._check_start_end_times  # check start and end times

        loader = self.return_loader(file_path=file_path,
                                    start=start, end=end,
                                    nchan=nchan, left_probe=left_probe,
                                    right_probe=right_probe)

        # ---------- loader summary
        loader_report_info(loader, common_reference=self.common_reference, nchan=nchan,
                           file_path=file_path, left_probe=left_probe, right_probe=right_probe)

        #  ---------- theshold calculation
        bpass_set = loader.get_traces()
        baseline_thresholds = find_mua_thresh(voltage_set=bpass_set,
                                              thresh_detect_mode=self.thresh_detect_mode,
                                              upthr=self.upthr)

        # ---------- mua spikes
        mua_spikes = extract_mua_spikes(voltage_set=bpass_set, chans=loader.get_channel_ids(),
                                        mua_thresh_set=baseline_thresholds, srate=self.srate)

        return [isolated_metadata, mua_spikes, bpass_set, loader.get_channel_ids(),
                loader.get_channel_groups(), baseline_thresholds]
