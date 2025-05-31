#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=TestTDMPC
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_output_%A.txt
#SBATCH --error=logs/slurm_error_%A.txt

# Load necessary modules (adjust based on your environment)
module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a

source tdmpc-env/bin/activate  # activate your virtual environment

LOGDIR=checkpoints

# Run the code
srun python src/test.py task=$1 reward=$2 checkpoint_dir=$LOGDIR/$3
