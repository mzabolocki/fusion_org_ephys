"""
Traces_extraction.py 

Marthe Priouret, priouretmarthe@orange.fr

Script to extract spatio-temporal components of firing neurons from Ca-imaging recordings.
This pipeline is based on Caiman, and requires prior installation of the Caiman environment before running. 
See documentation here: https://github.com/flatironinstitute/CaImAn.

This adaptation of the Caiman code was done in collaboration with Charlie Fieseler and Marielle Piber. 

"""


## Parameters to change if running a new video:
# - fnames
# - min_corr (default: .8)
# - min_pnr (default: 10)
# - min_SNR (default: 3)
# - r_values_min (default: 0.85)


import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os
import pickle
from caiman.utils import visualization
from matplotlib import pyplot as plt

logging.basicConfig(format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",level=logging.DEBUG)

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr, nb_inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
import cv2
from pathlib import Path

try:
    cv2.setNumThreads(0)
except:
    pass
import bokeh.plotting as bpl
import holoviews as hv

###################################################################################################
###################################################################################################


# Defines a function that will loop caiman through a whole folder
def analyze_full_folder_using_caiman(folder_name, recursion_depth=0, max_recursion=1):    
    for file in Path(folder_name).iterdir():
        if recursion_depth < max_recursion and file.is_dir():
            # Run this function recursively
            analyze_full_folder_using_caiman(file, recursion_depth=recursion_depth + 1, max_recursion=max_recursion)
        elif str(file).endswith('.tif'):
            print(f"Found tif file: {file}")
            analyze_single_file_using_caiman(file)
        else:
            print(f"Found non-tif file: {file}")

            
# Defines a function that will analyse a file: this is the caiman code put into one big function
def analyze_single_file_using_caiman(fname):
    # Check that the file exists
    assert Path(fname).exists(), "File not found"
    assert Path(fname).is_absolute(), "File path must be absolute"
    
    # name the outputs of caiman according to the file name of the file that is currently processed
    raw_caiman_fname = add_name_suffix(fname, '_raw_caiman')
    raw_caiman_fname = str(Path(raw_caiman_fname).with_suffix('.hdf5'))
    traces_fname = add_name_suffix(fname, '_traces')
    traces_fname = Path(traces_fname).with_suffix('.pickle')
    example_plot_fname = add_name_suffix(fname, '_example_plot')
    example_plot_fname = Path(example_plot_fname).with_suffix('.png')
    grid_plot_fname = add_name_suffix(fname, '_grid_plot')
    grid_plot_fname = Path(grid_plot_fname).with_suffix('.png')
    
    # Make a file in the same folder as this dataset, but with the same name for all datasets
    metadata_fname = Path(fname).with_name("metadata.csv")
    
    print(f"Producing output files (may take a while):")
    print(f"{raw_caiman_fname}, {traces_fname}, {example_plot_fname}, {grid_plot_fname}")
    
    fnames = [fname]


    # start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)


    # Setup some parameters

    # dataset dependent parameters
    frate = 65                       # movie frame rate
    decay_time = 0.4                 # length of a typical transient in seconds

    # motion correction parameters
    motion_correct = True    # flag for performing motion correction
    pw_rigid = False         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
    gSig_filt = (3, 3)       # size of high pass spatial filtering, used in 1p data
    max_shifts = (5, 5)      # maximum allowed rigid shift
    strides = (48, 48)       # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)      # overlap between pathes (size of patch strides+overlaps)
    max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'      # replicate values along the boundaries

    mc_dict = {
        'fnames': fnames,
        'fr': frate,
        'decay_time': decay_time,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }

    opts = params.CNMFParams(params_dict=mc_dict)


    # Motion Correction

    if motion_correct:
        # do motion correction rigid
        mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
        mc.motion_correct(save_movie=True)
        fname_mc1 = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
        if pw_rigid:
            bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                     np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
        else:
            bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)
        plt.subplot(1, 2, 1); plt.imshow(mc.total_template_rig)  # % plot template
        plt.subplot(1, 2, 2); plt.plot(mc.shifts_rig)  # % plot rigid shifts
        plt.legend(['x shifts', 'y shifts'])
        plt.xlabel('frames')
        plt.ylabel('pixels')

        bord_px = 0 if border_nan == 'copy' else bord_px
        fname_new = cm.save_memmap(fname_mc1, base_name='memmap_1_', order='C',
                               border_to_0=bord_px)
    else:  # if no motion correction just memory map the file
        fname_new = cm.save_memmap(fnames, base_name='memmap_',
                               order='C', border_to_0=0, dview=dview)


    # Load memory mapped file

    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order='F')


    # Parameter setting for CNMF-E
    # We now define some parameters for the source extraction step using the CNMF-E algorithm. 

    p = 1               # order of the autoregressive system
    K = 40            # upper bound on number of components per patch, in general None
    gSig = (4, 4)       # gaussian width of a 2D gaussian kernel, which approximates a neuron
    gSiz = (17, 17)     # average diameter of a neuron, in general 4*gSig+1
    Ain = None          # possibility to seed with predetermined binary masks
    merge_thr = .7      # merging threshold, max correlation allowed
    rf = 40             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
    stride_cnmf = 20    # amount of overlap between the patches in pixels
    #                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
    tsub = 2            # downsampling factor in time for initialization,
    #                     increase if you have memory problems # ARE WE DOWNSAMPLING EVEN MORE HERE???
    ssub = 1            # downsampling factor in space for initialization,
    #                     increase if you have memory problems
    #                     you can pass them here as boolean vectors
    low_rank_background = None  # None leaves background of each patch intact,
    #                     True performs global low-rank approximation if gnb>0
    gnb = 0             # number of background components (rank) if positive,
    #                     else exact ring model with following settings
    #                         gnb= 0: Return background as b and W
    #                         gnb=-1: Return full rank background B
    #                         gnb<-1: Don't return background
    nb_patch = 0        # number of background components (rank) per patch if gnb>0,
    #                     else it is set automatically
    min_corr = 0.4 #.95       # min peak value from correlation image #CRITICAL PARAMETER 
    min_pnr = 2.5 #10        # min peak to noise ration from PNR image #CRITICAL PARAMETER
    ssub_B = 2          # additional downsampling factor in space for background
    ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor

    opts.change_params(params_dict={'method_init': 'corr_pnr',  # use this for 1 photon
                                'K': K,
                                'gSig': gSig,
                                'gSiz': gSiz,
                                'merge_thr': merge_thr,
                                'p': p,
                                'tsub': tsub,
                                'ssub': ssub,
                                'rf': rf,
                                'stride': stride_cnmf,
                                'only_init': True,    # set it to True to run CNMF-E
                                'nb': gnb,
                                'nb_patch': nb_patch,
                                'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
                                'low_rank_background': low_rank_background,
                                'update_background_components': True,  # sometimes setting to False improve the results
                                'min_corr': min_corr,
                                'min_pnr': min_pnr,
                                'normalize_init': False,               # just leave as is
                                'center_psf': True,                    # leave as is for 1 photon
                                'ssub_B': ssub_B,
                                'ring_size_factor': ring_size_factor,
                                'del_duplicates': True,                # whether to remove duplicates from initialization
                                'border_pix': bord_px})                # number of pixels to not consider in the borders)

    # Inspect summary images and set parameters

    # compute some summary images (correlation and peak to noise)
    cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=gSig[0], swap_dim=False) 
    # inspect the summary images and set the parameters
    nb_inspect_correlation_pnr(cn_filter, pnr)


    # Run the CNMF-E algorithm

    cnm1 = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
    cnm1.fit(images)

    # Component Evaluation
    # The processing in patches creates several spurious components. These are filtered out by evaluating each component using three different criteria:
    # - the shape of each component must be correlated with the data at the corresponding location within the FOV
    # - a minimum peak SNR is required over the length of a transient
    # - each shape passes a CNN based classifier

    min_SNR = 5            # adaptive way to set threshold on the transient size
    r_values_min = 0.97    # threshold on space consistency (if you lower more components will be accepted, potentially with worst quality)
    cnm1.params.set('quality', {'min_SNR': min_SNR,
                           'rval_thr': r_values_min,
                           'use_cnn': False})
    cnm1.estimates.evaluate_components(images, cnm1.params, dview=dview)
    
    total_components = len(cnm1.estimates.C)
    accepted_components = len(cnm1.estimates.idx_components)
    append_metadata_to_saved_dataframe(fname, total_components, accepted_components, metadata_fname)

    print(' ***** ')
    print('Number of total components: ', total_components)
    print('Number of accepted components: ', accepted_components)
    
    # Defining a list of variables potentially useful for later
    idx1=cnm1.estimates.idx_components #this is so we have a list of the neurons that are accepted with our given criteria
    A1=cnm1.estimates.A  #this used the indexes (above) to select out the positional information of all accepted components
    C1=cnm1.estimates.C
    dims=dims #this variable contains the dimensions of our FOV usually 1042 x1042
    T1=cn_filter #this variable contains the "image" of our organoid, for the masks. (dont know if this is the correct one)
    
    save_results = True
    if save_results:
        cnm1.save(raw_caiman_fname)

    # Stop cluster
    cm.stop_server(dview=dview)

    # Extracting the accepted components for later processing
    matrix_of_traces=C1[idx1]

    # Saving what we need in a pickle file
    with open(traces_fname, 'wb') as outfile:
        pickle.dump(matrix_of_traces, outfile) #dumping all parameters into the pickle file. matrix_of_traces is the object written for this traces_fname file.

    # Plots: heatplot of all activity traces over time + individual neuronal traces
    fig, ax = plt.subplots(figsize=(25,15))
    ax.set_title('GCaMP1 ALL neurons over time')
    ax.set(xlabel="timepoints", ylabel="number of neurons")
    plt.pcolormesh(matrix_of_traces)
    plt.colorbar(label="Calcium intensity", pad=0.05)
    plt.savefig(example_plot_fname)

    with open(traces_fname, 'rb') as f:
        matrix_of_traces = pickle.load(f)
    
    # concatenating
    num_neurons = matrix_of_traces.shape[0]
    matrix_of_traces = np.array([np.hstack(matrix_of_traces[i_neuron, :]) for i_neuron in range(num_neurons)])
    print(matrix_of_traces.shape) 
    
    grid_plot(matrix_of_traces)
    plt.savefig(grid_plot_fname)
    
    
# Function to make a grid plot
def grid_plot(matrix_of_traces, num_columns=None, to_sharey=False, base_title='Neuron', start_of_counter=1):  
    # If nb of neurons < nb of columns, then 'AxesSubplot' object is not iterable
    num_neurons = matrix_of_traces.shape[0]
    if num_columns is None:
        num_columns = 5
    if num_neurons <= 5:
        num_columns = 2 #could we try to have 2 columns then or will this crash?
        print("Few neurons found, reducing number of columns")
    if num_neurons == 1:
        plt.plot(matrix_of_traces)
        plt.title("Only neuron found")
        return
    if num_neurons == 0:
        print("No neuron found")
        return
    num_rows = (num_neurons // num_columns) +1  
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(45, 45), sharex=True, sharey=to_sharey)
    axes = [item for items in axes for item in items]

    for i, ax in enumerate(axes):
        if i >= matrix_of_traces.shape[0]:
            break
        y = matrix_of_traces[i, :]
        title_str = f"{base_title}{i + start_of_counter}"
        ax.plot(y)
        ax.set_title(title_str, {'fontsize': 28}, y=0.7)
        ax.set_frame_on(False)
        ax.set_axis_off()

        
# Function to add a suffix to a name so that the output files' name is dependent on the input file name
def add_name_suffix(path: str, suffix='-1'):
    fpath = Path(path)
    base_fname, suffix_fname = fpath.stem, fpath.suffix
    new_base_fname = str(base_fname) + f"{suffix}"
    candidate_fname = fpath.with_name(new_base_fname + str(suffix_fname))

    return candidate_fname


# Function to write a text file with some metadata info
def append_metadata_to_saved_dataframe(fname, total_components, accepted_components, metadata_fname):
    # Check if file already exists. If so, load it and we will add to it
    if Path(metadata_fname).exists():
        df_old = pd.read_csv(metadata_fname)
    else:
        df_old = None
        
    # Create the object that has the new metadata
    metadata_dict = dict(total_components=[total_components],
                         accepted_components=[accepted_components],
                         original_filename=[fname])
    df_new = pd.DataFrame(metadata_dict)
    
    # Actually add to the old file
    if df_old is not None:
        list_of_dfs = [df_new, df_old]
        df_old = pd.concat(list_of_dfs, ignore_index=True)
    else:
        df_old = df_new
    
    # Save it, overwriting the old one
    df_old.to_csv(metadata_fname, index=False)

# Lines to run this code   
if __name__ == "__main__":
    
    # Design: only allow user to pass a full folder
    # Usage: python path/to/this/file.py --folder_name /absolute/path/to/data/folder/
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str)
    args = parser.parse_args()
    
    folder_name = args.folder_name
    analyze_full_folder_using_caiman(folder_name)
    
    print("Finished!")