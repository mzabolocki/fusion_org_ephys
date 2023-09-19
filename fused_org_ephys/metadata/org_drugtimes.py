"""
org_drug_times.py

Finds drug application times from
Open Ephys Message_Center-904.0 dir.

Return drug times from metadata dict
for select drug applications.

"""

import pathlib
import os
import math

import pandas as pd
import numpy as np


###################################################################################################
###################################################################################################


def attach_drugtimes(metadata=None, **syncmessage_kwargs):
    """
    Reads sync_messages.txt and timestamps.npy for drug times
    from the location of subfolders within the binary file acquired with
    open-ephys gui.

    All  drug times and txt files stored with respective drug_time_file_suffix
    and drug_text_file_suffix are automatically fetched.

    Args:
    -----
    metadata_df (df):
        organoid metadata df containing experiment date
        ('exp_days') and embryoid body formation ('EBs').

    **syncmessage_kwargs:
        sync message fpath key arguments.

    Returns:
    --------
    Attach drug txt and times to input metadata df.

    Notes:
    ------
    OpenEPhys GUI saves timestamps and txt files from aquisition
    in .../Message_Center-904.0/TEXT_group_1 folders.

    Sync processor delays are to be accounted for
    and adjusted in the *.npy files prior to loading.

    A simple script has been formulated to adjust for processor
    delays and re-save ('analysis/misc/drugapp_npy_changes/..')
    """

    syncmessage_texts = {'1_syncmessage_txt': [], '2_syncmessage_txt': [], '3_syncmessage_txt': [], '4_syncmessage_txt': []}
    syncmessage_times = {'1_syncmessage_time': [], '2_syncmessage_time': [], '3_syncmessage_time': [], '4_syncmessage_time': []}

    # path to drug txt + timestamps
    for file, exp, rec, record_node in zip(metadata['file'], metadata['experiment'],
                                           metadata['recording'], metadata['record_node']):

        # ----- open ephys GUI fpath
        if math.isnan(record_node) is True:
            syncmessage_folder = pathlib.PurePath(syncmessage_kwargs['base_folder'], file, 'experiment'+str(int(exp)),
                                                  'recording' + str(int(rec)), 'events', str('Message_Center-904.0'),
                                                  'TEXT_group_1')

        # ----- new open ephys GUI fpath
        else:
            syncmessage_folder = pathlib.PurePath(syncmessage_kwargs['base_folder'], file,
                                                  'Record Node '+str(int(record_node)),
                                                  'experiment'+str(int(exp)),
                                                  'recording' + str(int(rec)), 'events',
                                                  'Message_Center-904.0', 'TEXT_group_1')

        syncmessage_time_path = (str(syncmessage_folder) + '/' + syncmessage_kwargs['drug_time_file_suffix'] + '.npy')
        syncmessage_text_path = (str(syncmessage_folder) + '/' + syncmessage_kwargs['drug_text_file_suffix'] + '.npy')

        # ------ load timestamps
        if os.path.exists(syncmessage_time_path):
            syncmessage_time = np.load(syncmessage_time_path)
        else:
            syncmessage_time = None

        # ------ load text
        if os.path.exists(syncmessage_text_path):
            syncmessage_text = np.load(syncmessage_text_path)
        else:
            syncmessage_text = None

        # ------- add drug times + txt
        if all(v is not None for v in [syncmessage_text, syncmessage_time]):
            for idx in range(len(syncmessage_texts)):
                try:
                    syncmessage_texts[list(syncmessage_texts.keys())[idx]].append(syncmessage_text[idx])
                except IndexError:
                    syncmessage_texts[list(syncmessage_texts.keys())[idx]].append(np.nan)

                try:
                    syncmessage_times[list(syncmessage_times.keys())[idx]].append(syncmessage_time[idx])
                except IndexError:
                    syncmessage_times[list(syncmessage_times.keys())[idx]].append(np.nan)
        else:
            for txt_key, time_keys in zip(syncmessage_texts.keys(), syncmessage_times.keys()):
                syncmessage_texts[txt_key].append(np.nan)
                syncmessage_times[time_keys].append(np.nan)

    # -------- + metadata
    for text_key, times_key in zip(syncmessage_texts.keys(), syncmessage_times.keys()):
        metadata[times_key] = syncmessage_times[times_key]
        metadata[text_key] = syncmessage_texts[text_key]

    return pd.DataFrame(metadata)
