#!/bin/bash

echo "SLURM_VISIBLE_DEVICES: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

DATA="/home/lfranceschi/repos/ACL-SSL"

cd $DATA

python --version

python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 12345 \
myutils/test_env.py \
--model_name ACL_ViT16 \
--model_path $DATA/pretrain \
--exp_name aclifa_1gpu \
--train_config Exp_ACL_v2 \
--vggss_path $DATA/VGGSS \
--flickr_path $DATA/Flickr \
--avs_path $DATA/AVSBench/AVS1 \
--vggsound_path $DATA/vggsound \
--save_path $DATA \
--san
