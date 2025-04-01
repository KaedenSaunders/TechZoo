#!/bin/bash
#SBATCH --job-name=test_python_environment   # Descriptive job name
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --ntasks=1                           # Number of tasks per node
#SBATCH --time=24:00:00                      # Time limit (HH:MM:SS)
#SBATCH --gpus-per-node=1                    # Request a node with 1 GPU

# Variables.
export PROJECT_PATH=/home/ks679318/TechZoo

cd $PROJECT_PATH
source .venv/bin/activate
cd src

python train_dqn.py 
