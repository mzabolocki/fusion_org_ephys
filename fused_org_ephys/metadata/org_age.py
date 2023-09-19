"""
org_age.py

Module to calculate organoid
age from EB and recording dates. 

"""

import pandas as pd
from datetime import datetime


###################################################################################################
###################################################################################################


def compute_age(metadata_df):
    """
    Compute organoid ages. Subtract datetime of recording ('exp_days')
    from embryoid body formation ('EBs').

    Args:
    -----
    metadata_df (df):
        organoid metadata df containing experiment date
        ('exp_days') and embryoid body formation ('EBs').

    Returns:
    --------
    ages_set (list): list of organoid ages.
    """

    # data conversions
    # https://stackoverflow.com/questions/70884910/converting-dates-into-a-specific-format-in-side-a-csv
    exp_days = pd.to_datetime(metadata_df.exp_day, dayfirst=True).astype(str).tolist()
    exp_days = [s + "_00-00-00" for s in exp_days]
    EBs = pd.to_datetime(metadata_df['EB'], dayfirst=True).astype(str).tolist()
    EBs = [s + "_00-00-00" for s in EBs]

    ages_set = []
    for date, birth in zip(exp_days, EBs):
        if date is None or birth is None:
            age = 0

        else:

            # ----- recording date
            try:
                myformat = '%Y-%m-%d_%H-%M-%S'
                recdate = datetime.strptime(date, myformat)
            except ValueError:  # if not hour:min:sec precision
                recdate = datetime.strptime(date, '%Y-%m-%d')

            # ----- embryoid body date
            try:
                birthdate = datetime.strptime(birth, myformat)
            except ValueError:  # if not hour:min:sec precision
                birthdate = datetime.strptime(birth, '%Y-%m-%d')

            delta = recdate-birthdate
            age = delta.days + delta.seconds/(24*60*60)

            ages_set.append(age)

    return ages_set