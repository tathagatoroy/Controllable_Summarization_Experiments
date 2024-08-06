#!/bin/bash
cd .. 
export CUDA_VISIBLE_DEVICES=0
echo "running on gpu 0"
python  cascaded_lora_inference.py --model_path  raw_model.pth > ./logs/output1.txt &

export CUDA_VISIBLE_DEVICES=1
echo "running on gpu 1"
python cascaded_lora_inference.py --model_path model_first_attribute_after_100_steps.pth  > ./logs/output2.txt &

export CUDA_VISIBLE_DEVICES=2
echo "running on gpu 2"
python cascaded_lora_inference.py --model_path model_first_attribute_after_300_steps.pth  > ./logs/output3.txt &

export CUDA_VISIBLE_DEVICES=3
echo "running on gpu 3"
python cascaded_lora_inference.py --model_path model_second_attribute_after_100_steps.pth  > ./logs/output4.txt &