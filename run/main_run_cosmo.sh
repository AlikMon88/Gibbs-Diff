#!/bin/bash
#SBATCH --job-name=gdiff-cosmo
#SBATCH --output=/home/am3353/Gibbs-Diff/logs/gdiff_cosmo_output.log
#SBATCH --error=/home/am3353/Gibbs-Diff/logs/gdiff_cosmo_error.log
#SBATCH --time=20:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -A MPHIL-DIS-SL2-GPU

# Load necessary modules
module load python/3.9.12
module load cuda/11.8

# Activate your Python virtual environment
source /home/am3353/Gibbs-Diff/gdiff-env-csd3/bin/activate

# Execute the script directly with Python
# The -m flag runs the specified module as a script

accelerate launch -m --num_machines=1 --num_processes=1 modules.main_run --mode=cosmo
# accelerate launch -m modules.main_run --mode=cosmo
# python -m modules.main_run --mode=cosmo