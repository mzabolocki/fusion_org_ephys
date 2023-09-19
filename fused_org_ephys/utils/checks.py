"""
checks.py

Functions for sanity checks.

"""

###################################################################################################
###################################################################################################


def _check_frames(si_recording, start=None, end=None):
    """
    Checks if the frames are in the correct order.

    Parameters
    ----------
    si_recording : SIRecording
        loader object.
    start : int, optional
        Start frame. The default is None.
    end : int, optional
        End frame. The default is None.

    Returns
    -------
    None.
    """

    rec_length = si_recording.get_num_frames()

    # check start and end frames
    try:
        start_frame = int(start)
        end_frame = int(end)
    except TypeError:
        start_frame = int(start)
        end_frame = rec_length

    return start_frame, end_frame


def _check_reclength(si_recording, file, start=None, end=None):
    """
    Checks if the recording length is correct.

    Parameters
    ----------
    si_recording : SIRecording
        loader object.
    file : str
        file name.
    start : int, optional
        Start frame. The default is None.
    end : int, optional
        End frame. The default is None.

    Returns
    -------
    None.
    """

    end_frame = end
    start_frame = start

    # check end frame
    if end is not None:
        rec_length = si_recording.get_num_frames()
        if end > rec_length:
            print(f'end frame > total length for file {file}| analysing frames {start}:{rec_length}')
            end_frame = rec_length  # default to end
    else:
        pass

    return start_frame, end_frame
