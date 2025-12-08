#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1"
export OMP_NUM_THREADS="4"

python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port 12345 \
Train_ACL.py \
--model_name ACL_ViT16 \
--train_config Exp_ACL_v1 \
--exp_name aclifa_2gpu \
--vggss_path $DATA/VGGSS \
--flickr_path $DATA/Flickr \
--avs_path $DATA/AVSBench/AVS1 \
--save_path ""
