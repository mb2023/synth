#!/bin/bash

#SBATCH --job-name=synthmed_job
#SBATCH --output=synthmed_job_%j.out
#SBATCH --error=synthmed_job_%j.err
#SBATCH --time=01:00:00        # One hour runtime limit
#SBATCH --mem=8G               # Request 8 GB of memory
#SBATCH --cpus-per-task=1      # Request one CPU core

# Load the Conda environment to get Python and libraries
# Adjust the path to conda.sh if necessary for your HPC setup
source ~/.bashrc
conda activate synth_env

# Navigate to your project directory where the files were uploaded
cd /home/mb2023/my_hpc_projects/synthcity-docker-project

# Run the Python script
python generating.py