#! /bin/bash

# try to replicate the steps in the repo but with versions that will work, since nobody had the great idea to share an environment.yaml file

set -euo pipefail

module load conda

conda create -y -n acl_ssl27
conda activate acl_ssl27

conda install -y cudatoolkit
conda install -y -c conda-forge cudnn
conda install -y python=3.10
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0 numpy==1.24.1 --extra-index-url https://download.pytorch.org/whl/cu118
pip install tensorboard==2.11.2
pip install transformers==4.25.1
pip install opencv-python==4.7.0.72
pip install tqdm==4.65.0
pip install scikit-learn==1.2.2
pip install six==1.16.0

conda export --file environment.yaml