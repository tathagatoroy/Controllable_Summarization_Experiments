#!/bin/bash 

source /home2/tathagato/miniconda3/bin/activate roy

#export only gpu 0
export CUDA_VISIBLE_DEVICES=2

cd ..
#extractiveness then length on length


# #extractiveness then length on extractiveness
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/new_adapter_experiments/extractiveness \
# --second_checkpoint /scratch/tathagato/new_adapter_experiments/extractiveness_then_length/length \
# --output_file  ./results/extractiveness_then_length_evaluate_extractiveness.json \
# --attribute "extractiveness" \
# --first_adapter_name "extractiveness" \
# --second_adapter_name "length"


# #topic then length on topic
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/new_adapter_experiments/topic \
# --second_checkpoint /scratch/tathagato/new_adapter_experiments/topic_then_length/length \
# --output_file  ./results/topic_then_length_evaluate_topic.json \
# --attribute "topic" \
# --first_adapter_name "topic" \
# --second_adapter_name "length"

# #topic then length on length
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/new_adapter_experiments/topic \
# --second_checkpoint /scratch/tathagato/new_adapter_experiments/topic_then_length/length \
# --output_file  ./results/topic_then_length_evaluate_length.json \
# --attribute "length" \
# --first_adapter_name "topic" \
# --second_adapter_name "length"

# Dual attributes
# length then extractiveness on length
python do_inference.py \
--use_merged_model_checkpoint \
--merged_model_directory /scratch/tathagato/adapter_experiments/length_then_extractiveness \
--output_file ./merged_adapter_results/length_then_extractiveness_evaluate_length.json \
--attribute length

# length then extractiveness on extractiveness
python do_inference.py \
--use_merged_model_checkpoint \
--merged_model_directory /scratch/tathagato/adapter_experiments/length_then_extractiveness \
--output_file ./merged_adapter_results/length_then_extractiveness_evaluate_extractiveness.json \
--attribute extractiveness

# extractiveness then length on length
python do_inference.py \
--use_merged_model_checkpoint \
--merged_model_directory /scratch/tathagato/adapter_experiments/extractiveness_then_length \
--output_file ./merged_adapter_results/extractiveness_then_length_evaluate_length.json \
--attribute length

# extractiveness then length on extractiveness
python do_inference.py \
--use_merged_model_checkpoint \
--merged_model_directory /scratch/tathagato/adapter_experiments/extractiveness_then_length \
--output_file ./merged_adapter_results/extractiveness_then_length_evaluate_extractiveness.json \
--attribute extractiveness
