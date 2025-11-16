#!/bin/bash

# seems to be already set
# export CUDA_VISIBLE_DEVICES="0"

python Test_PTModels.py \
--model_name ACL_ViT16 \
--model_path $DATA/pretrain \
--exp_name test_best_param \
--vggss_path $DATA/VGGSS \
--flickr_path $DATA/Flickr \
--avs_path $DATA/Flickr \
--save_path "" \
--epochs None
