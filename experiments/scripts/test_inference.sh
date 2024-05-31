#!/bin/bash 
export CUDA_VISIBLE_DEVICES=0
source /home2/tathagato/miniconda3/bin/activate roy
cd ..
#single attribute
python do_inference.py --use_checkpoint \
--model_directory /scratch/tathagato/adapter_experiments/length/length \
--output_file  unnmerged_adapter_results/full_length_on_length.json \
--first_adapter_name "length" \
--attribute length
#/scratch/tathagato/adapter_experiments/length/checkpoint-400/length
python do_inference.py --use_checkpoint \
--model_directory /scratch/tathagato/adapter_experiments/length/checkpoint-400/length \
--output_file  unnmerged_adapter_results/400_length_on_length.json \
--first_adapter_name "length" \
--attribute length


python do_inference.py --use_checkpoint \
--model_directory /scratch/tathagato/adapter_experiments/length/checkpoint-800/length \
--output_file  unnmerged_adapter_results/800_length_on_length.json \
--first_adapter_name "length" \
--attribute length

python do_inference.py --use_checkpoint \
--model_directory /scratch/tathagato/adapter_experiments/length/checkpoint-1200/length \
--output_file  unnmerged_adapter_results/1200_length_on_length.json \
--first_adapter_name "length" \
--attribute length



