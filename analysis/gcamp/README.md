## gCAMP analysis notebook
-----

gCamp recordings were initially processed using [CaImAn](https://github.com/flatironinstitute/CaImAn). Outputs were analysed using the following pipeline constructed by Daniel Reumann and Marthe Priouret. Further details can be found below with example notebooks. 

## 1) Neuronal traces extraction
To extract the neuronal traces of the calcium imaging recordings, we used [CaImAn](https://github.com/flatironinstitute/CaImAn). Calcium imaging recordings were first preprocessed using a FIJI script to meet CaImAn requirements. Given that one calcium imaging recording is ~6.5 minutes long, the resolution of the recordings was downsampled to 512 * 512 pixels to overcome the 4GB size limit of the code. The first 100 frames of each recording were removed to avoid artefacts of the fluorescent signal. The video format was converted to .tiff. 

>An adaptation of the CaImAn code, with parameters adapted to our recordings and storage setup, can be found [here](https://github.com/mzabolocki/fusion_org_ephys/tree/main/fused_org_ephys/caiman). Inputs are .tiff files and outputs are matrices of traces with the spatio-temporal coordinates of the firing neurons stored in .pickle files.

## 2) Calculation of different variables
Following extraction, the matrices of neuronal traces were subjected to a filtering and different variables were calculated.
>A script for preprocessing the extracted traces and calculating different variables on them can be found [here](https://github.com/mzabolocki/fusion_org_ephys/blob/main/analysis/gcamp/gcamp_detect.ipynb).
