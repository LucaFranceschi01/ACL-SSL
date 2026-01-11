#!/bin/bash

#SBATCH --job-name=acl-train
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 4G
#SBATCH --gres=gpu:2
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luca.franceschi01@estudiant.upf.edu

set -e

env_name="acl-ssl"

module load Anaconda3
source activate $env_name

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lfranceschi/.conda/envs/$env_name/lib

# rsync -avz --exclude-from=myutils/excluded-directories.txt ~/repos/ACL-SSL $SCRATCH

# cd $SCRATCH/ACL-SSL

./Distributed_Experiment.sh

# rsync -avz Visual_results $DATA
# rsync -avz Train_record $DATA