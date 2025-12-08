#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

python Train_ACL.py \
--model_name ACL_ViT16 \
--model_path $DATA/pretrain \
--exp_name aclifa_1gpu \
--train_config Exp_ACL_v1 \
--vggss_path $DATA/VGGSS \
--flickr_path $DATA/Flickr \
--avs_path $DATA/AVSBench/AVS1 \
--save_path ""
