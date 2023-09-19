"""
df_loader.py

Modules to import xlsx files
from a directory.

"""

import glob
import os
import pandas as pd

###################################################################################################
###################################################################################################


def load_df(load_folder=None, save_folder=None, fname=None, drug=None):
    """
    load_df(load_folder = None, save_folder = None, fname = None, drug = None)

    Loads xlsx files from a directory.

    Parameters
    ----------
    load_folder : str, optional
        Path to the directory containing the xlsx files. The default is None.
    save_folder : str, optional
        Path to the directory where the xlsx files will be saved. The default is None.
    fname : str, optional
        Name of the xlsx file. The default is None.
    drug : str, optional
        Name of the drug. The default is None.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe containing the data.
    """

    appended_df = []
    if load_folder is not None:
        if isinstance(load_folder, str):

            file_paths = glob.glob(load_folder + "/*.xlsx")  # collect all file paths for .xlsx formats

            for file in file_paths:
                # print(f'loading {file} ...')
                df = pd.read_excel(file)
                appended_df.append(df)

            appended_df = pd.concat(appended_df)

            # update drug name
            # in exported df
            if drug is not None:
                appended_df['drugs'] = drug
            else:
                pass

            if save_folder is not None:
                isExist = os.path.exists(save_folder)
                if not isExist:
                    os.makedirs(save_folder)
                    print(f"The new directory is created for {save_folder}")

                save_path = str(save_folder + '/' + fname + '.xlsx')
                print(f'saved to: {save_path}')
                appended_df.to_excel(save_path)
            else:
                pass

            return appended_df
        else:
            raise TypeError('pass a str file path to the load folder ...')

    else:
        raise ValueError(f'load folder is {load_folder} | pass a file path ...')