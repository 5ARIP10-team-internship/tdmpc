#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=ArrayTDMPC
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --array=1-2%2
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_output_%A_%a.txt
#SBATCH --error=logs/slurm_error_%A_%a.txt

# Load necessary modules (adjust based on your environment)
module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a

source tdmpc-env/bin/activate  # activate your virtual environment

# Define the directory where to save the models, and copy the job file to it
JOB_FILE=$HOME/tdmpc/array.sh
HPARAMS_FILE=$HOME/tdmpc/array_hyperparameters.txt
TASK_NAME=$(echo $1 | cut -d'-' -f1)
CFG_FILE=$HOME/tdmpc/cfgs/tasks/${TASK_NAME}.yaml

LOGDIR=checkpoints/array_${SLURM_ARRAY_JOB_ID}
LOGPATH=$HOME/tdmpc/${LOGDIR}

mkdir -p $LOGPATH
rsync $CFG_FILE $LOGPATH/
rsync $JOB_FILE $LOGPATH/
rsync $HPARAMS_FILE $LOGPATH/

# Run the code
srun python src/train.py task=$1 checkpoint_dir=$LOGDIR/experiment_${SLURM_ARRAY_TASK_ID} \
            $(head -$SLURM_ARRAY_TASK_ID $HPARAMS_FILE | tail -1)