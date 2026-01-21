#!/bin/bash

#SBATCH --job-name=train-single
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu 32G
#SBATCH --exclude="node019,node020,node021,node022,node023,node024,node025,node026"
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luca.franceschi01@estudiant.upf.edu

set -e

env_name="acl-ssl"

module load Anaconda3
source activate $env_name

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lfranceschi/.conda/envs/$env_name/lib

./scripts/2-single_gpu_train.sh
