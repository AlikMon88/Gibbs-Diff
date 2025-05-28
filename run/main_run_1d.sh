#!/bin/bash
#SBATCH --job-name=gdiff-1d
#SBATCH --output=/home/am3353/am3353/logs/gdiff_1d_output.log
#SBATCH --error=/home/am3353/am3353/logs/gdiff_1d_error.log
#SBATCH --time=06:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                     # <- add this to be explicit
#SBATCH -A MPHIL-DIS-SL2-GPU

module load python/3.9.12
module load cuda/11.8

source /home/am3353/am3353/gdiff-env/bin/activate

# Use accelerate launch to control number of machines and processes
accelerate launch --num_machines=1 --num_processes=1 modules/main_run.py --mode=1D