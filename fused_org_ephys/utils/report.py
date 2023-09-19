"""
report_summary.py

Utilities to create reports
and useful print outs.

"""


###################################################################################################
###################################################################################################


def loader_report_info(loader, **loader_kwargs):
    """
    Print out information about the loader.

    Parameters
    ----------
    loader : object
        Loader object.
    loader_kwargs : dict
        Keyword arguments for the loader.

    """

    # collect params
    channel_ids = loader.get_channel_ids()
    fs = loader.get_sampling_frequency()
    rec_length = loader.get_num_frames()/fs

    # param descriptions
    desc = {
        'file_path': 'Loading *.dat file from file path',
        'channel_ids': 'Initial number of channel IDs',
        'postqc_channels': 'Post-channel removal channel number (including AUX numbers)',
        'probe': 'Left & right probe selection',
        'fs': 'Sampling frequency (Hz)',
        'rec_len': 'Total recording length (sec.)',
        'threshold': 'Spike threshold parameters',
        'cmr': 'Common reference (median, mean or None)',
        }

    # create output string
    str_lst = [
        # header
        '=',
        '',
        '\N{brain} ORGANOID OPENEPHYS LOADER OVERVIEW \N{brain}',
        '',

        # file summary
        *[el for el in ['\n',
                        'File path : {}'.format(loader_kwargs['file_path']),
                        '{}'.format(desc['file_path']),
                        '\n',
                        'Left probe : {}'.format(loader_kwargs['left_probe']),
                        'Right probe : {}'.format(loader_kwargs['right_probe']),
                        '{}'.format(desc['probe']),
                        '\n',
                        'Channel id length : {}'.format(loader_kwargs['nchan']),
                        '{}'.format(desc['channel_ids']),
                        '\n',
                        'Post-QC channels removed : {}'.format(loader_kwargs['nchan'] - len(channel_ids)),
                        '{}'.format(desc['postqc_channels']),
                        '\n',
                        'Common reference : {}'.format(loader_kwargs['common_reference']),
                        '{}'.format(desc['cmr']),
                        '\n',
                        'Sampling frequency (Hz) : {}'.format(fs),
                        '{}'.format(desc['fs']),
                        '\n',
                        'Recording length (seconds): {}'.format(rec_length),
                        '{}'.format(desc['rec_len'])] if el != ''],

        # footer
        '',
        '='
    ]

    # Set centering value - use a smaller value if in concise mode
    concise = True
    center_val = 70

    # Expand the section markers to full width
    str_lst[0] = str_lst[0] * center_val
    str_lst[-1] = str_lst[-1] * center_val

    # Drop blank lines, if concise
    str_lst = list(filter(lambda x: x != '', str_lst)) if concise else str_lst

    # Convert list to a single string representation, centering each line
    output = '\n'.join([string.center(center_val) for string in str_lst])

    print(output)
