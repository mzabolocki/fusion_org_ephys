"""
save.py

Create directory and path for figure savings and
exporters.

"""

import os
import pickle as pkl

###################################################################################################
###################################################################################################


def save_pickle(data_arr=None, file=None, base_folder=None, save_folder=None):
    """
    Save pickle file.

    Parameters
    ----------
    data_arr : ndarray
        Data to be saved.
    file : str
        Name of file to be saved.
    base_folder : str
        Base folder to save file.
    save_folder : str
        Folder to save file.
    """

    if save_folder is not None:
        f_path = create_path(str(base_folder + save_folder),
                             str(file), '.pkl')

        print(f'saving to: {f_path}')

        with open(f_path, 'wb') as f:
            pkl.dump(data_arr, f)

        del data_arr  # delete

        print('saved')

    else:
        raise ValueError(f'set save folder path | save_folder: {save_folder}')


def create_path(folderpath, fname, extension):
    """
    Create path for saving files.

    Parameters
    ----------
    folderpath : str
        Path to save file.
    fname : str
        Name of file.
    extension : str
        File extension.

    Returns
    -------
    f_path : str
        Path to save figure.
    """

    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        print(f"The new directory is created for {folderpath}")
    else:
        print(f"The directory already exists for {folderpath}")

    fpath = folderpath + fname + extension  # original df before visual inspection

    return fpath


def save_fig(fig, figname, file_extension='.png', save_folder=None):
    """
    Save figure.

    Parameters
    ----------
    fig : plt.figure
        Figure to be saved.
    figname : str
        Name of figure.
    save_folder : str
        Folder to save figure.
    """

    if save_folder is not None:
        f_path = create_path(str(save_folder), str(figname), file_extension)

        print(f'saving to: {f_path}')

        fig.savefig(f_path, dpi=300, bbox_inches="tight")

        print('saved')


def save_df(df, fname, save_folder=None):
    """
    Save dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be saved.
    fname : str
        Name of dataframe.
    save_folder : str
        Folder to save dataframe.
    file_extension : str
        file extension to save dataframe. Default '.xlsx'.
    """

    if save_folder is not None:
        f_path = create_path(str(save_folder), str(fname), '.xlsx')

        print(f'saving to: {f_path}')

        df.to_excel(f_path)

        print('saved')

    else:
        raise ValueError(f'set save folder path | save_folder: {save_folder}')
