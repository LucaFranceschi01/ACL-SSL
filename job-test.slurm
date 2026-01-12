#!/bin/bash

#SBATCH --job-name=infer
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 3
#SBATCH --partition express
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

./Test_PTModels.sh
# srun Test_PTModels.sh

# copiar resultados a $DATA?
rsync -avz Visual_results $DATA