#!/bin/bash
# This is a Slurm batch script to run a Python program.

#SBATCH --job-name=generate_fake_data     # A descriptive name for your job
#SBATCH --output=generate_data_%j.out     # Standard output file (%j will be replaced by job ID)
#SBATCH --error=generate_data_%j.err      # Standard error file (%j will be replaced by job ID)
#SBATCH --nodes=1                         # Request 1 compute node
#SBATCH --ntasks=1                        # Request 1 CPU core (task)
#SBATCH --time=00:05:00                   # Maximum run time (HH:MM:SS) - 5 minutes should be ample
#SBATCH --mem=2G                          # Memory per node (e.g., 2 Gigabytes) - adjust if your data grows much larger

# --- Environment Setup ---
source ~/.bashrc # Or ~/.bash_profile, ~/.profile depending on your shell setup

# Activate your Conda environment
conda activate synth_env

# --- Error Handling for Environment Activation ---
# It's good practice to check if the environment activation was successful.
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate Conda environment 'synth_env'."
    echo "Please check if the environment exists and 'conda activate' works in your shell."
    exit 1 # Exit the job if activation fails
fi

# --- Navigate to your project directory ---
cd /home/diya/my_project/synthcity-docker-project/ # Adjust this to your actual path on the HPC

# --- Execute your Python script ---
echo "Starting Python script: generate_data.py"
python generate_data.py

echo "Python script finished."