#!/bin/bash

"""
Caiman_jobarray.sbatch
Marthe Priouret, priouretmarthe@orange.fr
This is a batch script designed to run the script "Traces_extraction.py" on several subfolders at a time on an institute cluster to minimize running time.
The 'folder_name' entry needs to be modified before usage. The script should run in the CAIMAN environment.
Terminal command lines:
$ conda activate caiman
$ cd TSC_organoid_ephys/util/gcamp_processing/caiman
$ sbatch --export=ALL,Target='folder_name' Caiman_jobarray.sbatch
""" 

# === SLURM directives ===
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=30G # We will probably need more?
#SBATCH --time=24:00:00
#SBATCH --qos=medium # Put medium it we need between 8h and 2 days of walltime, rapid if less than 8h
#SBATCH --array=0-2 # need as many subtasks as they are subfolders within one batch
#SBATCH --output=jobarray_%A_%a_out.txt  #This setting will produce a text file for each of the job array subtask (%A is job name, %a is array number)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=marthe.priouret@imba.oeaw.ac.at


# === DATA SETUP ===
# The sbtach script goes through the "Targetsubfolder", searches through all subfolders with the key "GC" in the name
# and assign a as a job array each sub folder that contains the key in its name

echo $Target
array=()
while IFS= read -r -d $'\0'; do
    array+=("$REPLY")
    #echo $REPLY
done < <(find /groups/knoblich/Organoid_Research/TSC_aHead/Experiments/GCaMP_Recordings/$Target -type d -name *day* -print0)

declare -p array

# When the lenght of the array is bigger than the number of task ID (actual subfolders), then abort the mission
# if ["${#array[@]}" -lt "${SLURM_ARRAY_TASK_ID}"]; then
#     echo "lenght of array inferior to number of jobs"
#     exit
# fi 

# Get the SINGLE folder from this list that this job actually works on
folder_to_analyse=${array[${SLURM_ARRAY_TASK_ID}]}


# === DO THE WORK ===

# Prints out the command that we are running
COMMAND="/users/marthe.priouret/TSC_organoid_ephys/util/gcamp_processing/caiman/Traces_extraction.py --folder_name $folder_to_analyse"
echo "Running command:"
echo $COMMAND

# Invoke the "Caiman_jobarrays.py" script
START_TIME=${SECONDS}
python /users/marthe.priouret/TSC_organoid_ephys/util/gcamp_processing/caiman/Traces_extraction.py --folder_name "$folder_to_analyse" 
END_TIME=${SECONDS}

# Prints out the time it took
echo "Traces extraction of ${folder_to_analyse} took $((END_TIME - START_TIME)) seconds"