#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

DATA="/home/lfranceschi/repos/ACL-SSL"

python Train_ACL_on_vggsound.py \
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
