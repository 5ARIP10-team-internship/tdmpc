#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=TrainTDMPC
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output_%A.txt
#SBATCH --error=slurm_error_%A.txt

# Load necessary modules (adjust based on your environment)
module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a

source tdmpc-env/bin/activate  # activate your virtual environment

# Define the directory where to save the models, and copy the job file to it
JOB_FILE=$HOME/tdmpc/train_hpc.sh
TASK_NAME=$(echo $1 | cut -d'-' -f1)
HPARAMS_FILE=$HOME/tdmpc/cfgs/tasks/${TASK_NAME}.yaml

LOGDIR=logs/train_${SLURM_JOB_ID}
LOGPATH=$HOME/tdmpc/${LOGDIR}

mkdir $LOGPATH
rsync $HPARAMS_FILE $LOGPATH/
rsync $JOB_FILE $LOGPATH/

# Run the code
srun python src/train.py task=$1 checkpoint_dir=$LOGDIR/${TASK_NAME}