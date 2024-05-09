#!/bin/bash 
export CUDA_VISIBLE_DEVICES=0
source /home2/tathagato/miniconda3/bin/activate roy
cd ..
python run_inference.py  --attributes "length" "extractiveness" 