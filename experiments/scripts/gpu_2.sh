#!/bin/bash 

source /home2/tathagato/miniconda3/bin/activate roy

#export only gpu 0
export CUDA_VISIBLE_DEVICES=1

cd ..
# python do_inference.py --use_checkpoint \
# --model_directory /scratch/tathagato/new_adapter_experiments/length \
# --output_file  ./results/extractiveness_on_extractiveness.json \
# --first_adapter_name extractiveness \
# --attribute extractiveness

# python do_inference.py --use_checkpoint \
# --model_directory /scratch/tathagato/new_adapter_experiments/topic \
# --output_file  ./results/topic_on_topic.json \
# --first_adapter_name "topic" \
# --attribute topic

# #now dual on both attributes
# #length then extractiveness on length
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/new_adapter_experiments/length \
# --second_checkpoint /scratch/tathagato/new_adapter_experiments/length_then_extractiveness/extractiveness \
# --output_file  ./results/length_then_extractiveness_evaluate_length.json \
# --attribute length \
# --first_adapter_name "length" \
# --second_adapter_name "extractiveness"

# #length then extractiveness on extractiveness
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/new_adapter_experiments/length \
# --second_checkpoint /scratch/tathagato/new_adapter_experiments/length_then_extractiveness/extractiveness \
# --output_file  ./results/length_then_extractiveness_evaluate_extractiveness.json \
# --attribute "extractiveness" \
# --first_adapter_name "length" \
# --second_adapter_name "extractiveness"

# #extractiveness then length on length
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/new_adapter_experiments/extractiveness \
# --second_checkpoint /scratch/tathagato/new_adapter_experiments/extractiveness_then_length/length \
# --output_file  ./results/extractiveness_then_length_evaluate_length.json \
# --attribute "length" \
# --first_adapter_name "extractiveness" \
# --second_adapter_name "length"
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/new_adapter_experiments/extractiveness \
# --second_checkpoint /scratch/tathagato/new_adapter_experiments/length \
# --output_file ./results/extractiveness_and_length_evaluate_length.json \
# --attribute "length" \
# --first_adapter_name "extractiveness" \
# --second_adapter_name "length"

# length then topic on length
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/length_then_topic \
# --output_file ./merged_adapter_results/length_then_topic_evaluate_length.json \
# --attribute length

# # length then topic on topic
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/length_then_topic \
# --output_file ./merged_adapter_results/length_then_topic_evaluate_topic.json \
# --attribute topic

# # topic then length on length
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/topic_then_length \
# --output_file ./merged_adapter_results/topic_then_length_evaluate_length.json \
# --attribute length

# # topic then length on topic
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/topic_then_length \
# --output_file ./merged_adapter_results/topic_then_length_evaluate_topic.json \
# --attribute topic


python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/length/checkpoint-1000/length \
--attribute length \
--output_file ./inference_results/length_evaluate_on_length_1000.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/length/checkpoint-1500/length \
--attribute length \
--output_file ./inference_results/length_evaluate_on_length_1500.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/length/length/ \
--attribute length \
--output_file ./inference_results/length_evaluate_on_length_final.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/specificity/checkpoint-800/specificity \
--attribute specificity \
--output_file ./inference_results/specificity_evaluate_on_specificity_800.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/specificity/checkpoint-1200/specificity \
--attribute specificity \
--output_file ./inference_results/specificity_evaluate_on_specificity_1200.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi