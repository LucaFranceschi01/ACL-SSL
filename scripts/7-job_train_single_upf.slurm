#!/bin/bash

#SBATCH --job-name=acl-train
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luca.franceschi01@estudiant.upf.edu

set -e

env_name="acl-ssl"

module load Anaconda3
source activate $env_name

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lfranceschi/.conda/envs/$env_name/lib

./2-single_gpu_train.sh
