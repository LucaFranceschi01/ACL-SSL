#!/bin/bash

python Test_PTModels.py \
--model_name ACL_ViT16 \
--model_path $DATA/pretrain \
--exp_name test_best_param \
--vggss_path $DATA/VGGSS \
--flickr_path $DATA/Flickr \
--avs_path $DATA/AVSBench/AVS1 \
--save_path "" \
--epochs None
