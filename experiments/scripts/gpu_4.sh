#!/bin/bash 

source /home2/tathagato/miniconda3/bin/activate roy

#export only gpu 0
export CUDA_VISIBLE_DEVICES=3

cd ..
#topic then extractiveness on topic
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/new_adapter_experiments/topic \
# --second_checkpoint /scratch/tathagato/new_adapter_experiments/topic_then_extractiveness/extractiveness \
# --output_file  ./results/topic_then_extractiveness_evaluate_topic.json \
# --attribute "topic" \
# --first_adapter_name "topic" \
# --second_adapter_name "extractiveness"

# #topic then extractiveness on extractiveness
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/new_adapter_experiments/topic \
# --second_checkpoint /scratch/tathagato/new_adapter_experiments/topic_then_extractiveness/extractiveness \
# --output_file  ./results/topic_then_extractiveness_evaluate_extractiveness.json \
# --attribute "extractiveness" \
# --first_adapter_name "topic" \
# --second_adapter_name "extractiveness"

# #extractiveness then topic on topic
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/new_adapter_experiments/extractiveness \
# --second_checkpoint /scratch/tathagato/new_adapter_experiments/extractiveness_then_topic/topic \
# --output_file  ./results/extractiveness_then_topic_evaluate_topic.json \
# --attribute "topic" \
# --first_adapter_name "extractiveness" \
# --second_adapter_name "topic"

#zero shot
# python do_inference.py \
# --output_file  ./merged_adapter_results/zero_shot_length \
# --attribute length

# python do_inference.py \
# --output_file  ./merged_adapter_results/zero_shot_extractiveness \
# --attribute extractiveness

# python do_inference.py \
# --output_file  ./merged_adapter_results/zero_shot_topic \
# --attribute topic

# # Single attribute
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/length \
# --output_file ./merged_adapter_results/length_on_length.json \
# --attribute length

# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/extractiveness \
# --output_file ./merged_adapter_results/extractiveness_on_extractiveness.json \
# --attribute extractiveness

# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/topic \
# --output_file ./merged_adapter_results/topic_on_topic.json \
# --attribute topic


python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/extractiveness/checkpoint-1/extractiveness \
--attribute extractiveness \
--output_file ./inference_results/extractiveness_evaluate_on_extractiveness_1.json  
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/extractiveness/checkpoint-500/extractiveness \
--attribute extractiveness \
--output_file ./inference_results/extractiveness_evaluate_on_extractiveness_500.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/extractiveness/checkpoint-1000/extractiveness \
--attribute extractiveness \
--output_file ./inference_results/extractiveness_evaluate_on_extractiveness_1000.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi
  





python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/specificity/specificity \
--attribute specificity \
--output_file ./inference_results/specificity_evaluate_on_specificity_full.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/specificity/checkpoint-600/specificity \
--attribute specificity \
--output_file ./inference_results/specificity_evaluate_on_specificity_600.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi