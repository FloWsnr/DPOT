#!/usr/bin/bash

### Task name
#SBATCH --account=xxxxxxx
#SBATCH --job-name=train_gphyt

### Output file
#SBATCH --output=results/slrm_logs/train_gphyt_%j.out


### Start a parallel job for a distributed-memory system on several nodes
#SBATCH --nodes=1

### How many CPU cores to use
#SBATCH --ntasks-per-node=96
#SBATCH --exclusive

### Mail notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=email@example.com

### Maximum runtime per task
#SBATCH --time=72:00:00

### set number of GPUs per task
#SBATCH --gres=gpu:4

### create time series, i.e. 100 jobs one after another. Each runs for 24 hours
##SBATCH --array=1-10%1

#####################################################################################
############################# Setup #################################################
#####################################################################################

# activate conda environment
export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate gphyt

######################################################################################
############################# Set paths ##############################################
######################################################################################
# debug mode
# debug=true
sim_name="poseidon_test00"
# Set up paths
base_dir="/home/flwi01/coding/DPOT"
python_exec="${base_dir}/dpot/train_well.py"
checkpoint_path="${base_dir}/results/${sim_name}"
data_dir="/home/flwi01/coding/well_datasets"
config_file="${base_dir}/configs/pretrain_medium.yaml"
export OMP_NUM_THREADS=1 # (num cpu - num_workers) / num_gpus

# finetune:
# path="/home/flwi01/coding/poseidon/results/poseidon_test00/Large-Physics-Foundation-Model/poseidon_test00/checkpoint-200"


accelerate_args="
--config_file ./configs/accel_config.yaml \
--num_cpu_threads_per_process 8"


#####################################################################################
############################# Training GPM ##########################################
#####################################################################################

exec_args="--config $config_file --data_path $data_dir \
--checkpoint_path $checkpoint_path"

# Capture Python output and errors in a variable and run the script
accelerate launch $python_exec $exec_args
