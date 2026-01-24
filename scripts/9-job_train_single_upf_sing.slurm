#!/bin/bash

#SBATCH --job-name=acl-train2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 16G
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luca.franceschi01@estudiant.upf.edu

set -e

nvidia-smi

singularity run --nv --bind /home/lfranceschi/repos/ACL-SSL container.sif \
/home/lfranceschi/repos/ACL-SSL/scripts/2-single_gpu_train.sh