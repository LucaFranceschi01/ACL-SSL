#!/bin/bash

#SBATCH --job-name=train-test
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
#SBATCH --time=0-00:30:00

set -e

env_name="acl-ssl"

module load conda
conda activate $env_name

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DATA/.conda/envs/$env_name/lib

rsync -avz --exclude-from=myutils/excluded-directories.txt ~/repos/ACL-SSL $SCRATCH

cd $SCRATCH/ACL-SSL

./SingleGPU_Experiment.sh

rsync -avz Visual_results $DATA
rsync -avz Train_record $DATA