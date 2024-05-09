#!/bin/bash 
export CUDA_VISIBLE_DEVICES=1
source /home2/tathagato/miniconda3/bin/activate roy
cd ..
python qlora_train.py --attribute extractiveness --load_previous_model --previous_model_path "/scratch/tathagato/openelm_adapter experiments/2024-05-01-18-26-09_length/checkpoint-4278"