#!/bin/bash
#SBATCH --job-name=thaind       # Job name
#SBATCH --output=w_prestrain_res.txt      # Output file
#SBATCH --error=w_prestrain_error.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=4G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node

cd /home/user01/aiotlab/thaind/LLaVA
export PYTHONPATH=/home/user01/aiotlab/thaind/LLaVA:$PYTHONPATH
sh scripts/pretrain_1.sh