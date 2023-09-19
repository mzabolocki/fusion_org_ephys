"""
metadata_params.py

Modules for metadata params
for loader classes.
"""

import numpy as np


###################################################################################################
###################################################################################################


def _fileinfo(expIDs_df, file_path_info):
    """
    Get file info from file_path_info.

    Parameters
    ----------
    expIDs_df : pandas.DataFrame
        Experiment IDs dataframe.
    file_path_info : pandas.DataFrame
        File path info dataframe.

    Returns
    -------
    file_info : pandas.DataFrame
        File info dataframe.
    """

    expIDidx = expIDs_df.index.values

    nchans = np.array(file_path_info['nchans'])[expIDidx]
    files = np.array(file_path_info['files'])[expIDidx]
    file_paths = np.array(file_path_info['file_paths'])[expIDidx]

    return {'nchans': nchans, 'files': files, 'file_paths': file_paths}


def _processingtimes(expIDs_df, baseline_start_end_times, drug_start_end_times):
    """
    Get processing times from baseline_start_end_times and drug_start_end_times.

    Parameters
    ----------
    expIDs_df : pandas.DataFrame
        Experiment IDs dataframe.
    baseline_start_end_times : pandas.DataFrame
        Baseline start end times dataframe.
    drug_start_end_times : pandas.DataFrame
        Drug start end times dataframe.

    Returns
    -------
    processing_times : pandas.DataFrame
        Processing times dataframe.
    """

    expIDidx = expIDs_df.index.values

    baseline_start = np.array(baseline_start_end_times['start'])[expIDidx]
    baseline_end = np.array(baseline_start_end_times['end'])[expIDidx]
    drug_start = np.array(drug_start_end_times['start'])[expIDidx]
    drug_end = np.array(drug_start_end_times['end'])[expIDidx]

    return {'baseline_start': baseline_start, 'baseline_end': baseline_end,
            'drug_start': drug_start, 'drug_end': drug_end}


def _isolate_metadata(expIDs_df, file):
    """
    Get isolate metadata from file.

    Parameters
    ----------
    expIDs_df : pandas.DataFrame
        Experiment IDs dataframe.
    file : pandas.DataFrame
        File dataframe.

    Returns
    -------
    isolate_metadata : pandas.DataFrame
        Isolate metadata dataframe.
    """

    return expIDs_df[expIDs_df.file == file]
