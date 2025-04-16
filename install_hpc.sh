#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=InstallEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:15:00
#SBATCH --output=slurm_output_%A.txt
#SBATCH --error=slurm_error_%A.txt

# Load necessary modules (adjust based on your environment)
module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a

# Create and activate virtual environment
python -m venv tdmpc-env --system-site-packages
source tdmpc-env/bin/activate

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt

# Test GPU availability
srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"
