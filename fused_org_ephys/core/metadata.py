"""
metadata.py

Class for loading metadata TC metadata
for silicon probe extracellular recordings.

Note: this is designed specifically for the Knoblich Lab.
TC documentations should be adapted for specific lab purposes.
"""

import pathlib
import math
import pandas as pd
from ..metadata import (attach_drugtimes, compute_age)


###################################################################################################
###################################################################################################


class Metadata:
    """
    Metadata class - core object which contains all metadata.

    Automatically load and organize organoid tissue culture details.

    Integrate all recording params and drug application times for
    binary files recorded with Open Ephys using silicon probes
    from Cambridge Neurotech.

    Arguments:
    ----------
    metadata_path (str, *.xlsx file extension):
        path to tc metadata. Defaults to None.
    base_folder (str):
        base folder containing all binary files. Defaults to 'TSC2_Ephys_Data'.
    drug_time_file_suffix (str):
        suffix for timestamps.npy. Defaults to 'timestamps_update'.
    drug_text_file_suffix (str):
        suffix for sync_messages.txt. Defaults to 'text_update'.
    recording_segment_selection (str, int):
        recording segment selection ('baseline', 1, 2, 3 ...). Defaults to 'baseline'.

    Notes:
    ------
    OpenEPhys GUI saves timestamps and txt files from aquisition
    in .../Message_Center-904.0/TEXT_group_1 folders. These are
    used to extract optogenetic stimulation times and/or drug application times.

    Sync processor delays sometimes found with the Open Ephys GUI
    must be accounted for and adjusted in the *.npy files prior to initiating the class.
    """

    def __init__(self, metadata_path=None, base_folder='Daniel_SiPro_Data/Data',
                 drug_time_file_suffix='timestamps_update',
                 drug_text_file_suffix='text_update'):
        """Initialize object with desired settings."""

        self.read_drugtimes_path = {'drug_time_file_suffix': drug_time_file_suffix,
                                    'drug_text_file_suffix': drug_text_file_suffix,
                                    'base_folder': base_folder}

        self.metadata_path = metadata_path
        self.base_folder = base_folder
        self.metadata_tc_df = self.return_tcmetadata
        self.metadata = self.return_metadata
        self.file_path_info = self.return_file_path_info

    @property
    def return_tcmetadata(self):
        """
        Return tc metadata *.xlsx file.

        Raises:
        -------
        TypeError: no metadata path.

        Returns:
        -------
        metadata_tc_df (df):
            metadata df from tissue culture.

        Notes:
        ------
        Metadata contains organoid TC information and is specific
        to the knoblich lab. It does not contain drug times or org ages.
        """

        if (self.metadata_path is not None) and (self.metadata_path is not False):
            metadata_tc_df = pd.read_excel(self.metadata_path)
            return metadata_tc_df
        elif self.metadata_path is None:  # no metadata path found
            raise TypeError(f'no metadata path found | {self.metadata_path}')
        elif self.metadata_path is False:  # no metadata needed (e.g. for preprocessed data)
            return None

    @property
    def return_metadata(self):
        """
        Return metadata for all organoid recordings.

        Automatically attach computed organoid ages and drug application
        times and details from sync_messages.txt and timestamps.npy
        from the location of the binary file acquired with open-ephys gui.

        Returns:
        --------
        metadata_df (df):
            metadata df containing organoid recording details
            (i.e. drug application times, org ages, patient lines ...).
        """

        if self.metadata_tc_df is not None:

            metadata_col = ['expID', 'exp_day', 'file',
                            'EB', 'condition', 'left_probe', 'right_probe',
                            'organoid', 'record_node', 'experiment',
                            'recording', 'channel_map', 'intan_controller',
                            'nchan', 'drugs', 'drug_addition', 'probe_region',
                            'optostim_region', 'opto_start_1', 'opto_end_1']

            # discard organoids subject to cell-line abberations
            metadata_df = self.metadata_tc_df[self.metadata_tc_df.usable == 'yes'].reset_index()

            # filter for relevant metadata info
            metadata_df = metadata_df.filter(items=metadata_col).dropna(thresh=4).reset_index()

            # attach ages and drug application times
            # if all columns in metadata_col are present
            if all([item in metadata_df.columns for item in metadata_col]):  # turn off if not needed

                metadata_df['ages'] = compute_age(metadata_df)
                metadata_df = attach_drugtimes(metadata=metadata_df.to_dict('list'), **self.read_drugtimes_path)

                return metadata_df

            # error raised if not all columns are present
            # this was built as a sanity check for the analysis
            else:
                raise ValueError(f'metadata does not contain the following columns | {metadata_col}')

        return None

    @property
    def return_file_path_info(self):
        """
        Returns file paths for binary open ephys files.

        Returns:
        --------
        file_path_info (dict):
            file path informaton for loading binary open ephys files.

        Notes:
        ------
        Differences in fpaths are due to the added record node
        subdir with the latest open ephys gui update. See below for more details.

        https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/1923547141/RecordNode
        """

        if self.metadata is not None:

            file_path_info = {'file_paths': [], 'nchans': [], 'files': []}

            for index, row in self.metadata[['file', 'record_node', 'experiment', 'recording',
                                            'channel_map', 'intan_controller', 'nchan']].iterrows():

                # NOTE ::
                # the following is to deal with the fact that
                # the open ephys GUI has changed the way it saves files
                # and the way it saves the file path between versions.
                # this is a temporary fix until the open ephys GUI is updated.
                # see https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/1923547141/RecordNode

                # ----- open ephys GUI fpath
                if math.isnan(row['record_node']) is True:

                    file_path = pathlib.PurePath(self.base_folder, row['file'], 'experiment'+str(int(row['experiment'])),
                                                 'recording'+str(int(row['recording'])), 'continuous',
                                                 str('Channel_Map-')+row['channel_map'], 'continuous.dat')

                # ----- new open ephys GUI fpath
                else:
                    file_path = pathlib.PurePath(self.base_folder, row['file'],
                                                 'Record Node '+str(int(row['record_node'])),
                                                 'experiment'+str(int(row['experiment'])),
                                                 'recording'+str(int(row['recording'])), 'continuous',
                                                 'Intan_Rec._Controller-' + str(row['intan_controller']),
                                                 'continuous.dat')

                file_path_info['file_paths'].append(file_path)
                file_path_info['nchans'].append(row['nchan'])
                file_path_info['files'].append(row['file'])

            return file_path_info

        else:  # if no metadata is found, no need to return file paths
            return None
