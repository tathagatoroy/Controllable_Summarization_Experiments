#!/bin/bash 
export CUDA_VISIBLE_DEVICES=0
source /home2/tathagato/miniconda3/bin/activate roy
cd ..
python qlora_train.py --load_previous_model --previous_model_path "/scratch/tathagato/openelm_adapter experiments/2024-05-01-19-01-27_extractiveness/checkpoint-8556"