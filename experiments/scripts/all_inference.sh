#!/bin/bash


cd ..
#zero shot
python do_inference.py \
--output_file  ./merged_adapter_results/zero_shot_length \
--attribute length

python do_inference.py \
--output_file  ./merged_adapter_results/zero_shot_extractiveness \
--attribute extractiveness

python do_inference.py \
--output_file  ./merged_adapter_results/zero_shot_topic \
--attribute topic

# Single attribute
python do_inference.py \
--use_merged_model_checkpoint \
--merged_model_directory /scratch/tathagato/adapter_experiments/length \
--output_file ./merged_adapter_results/length_on_length.json \
--attribute length

python do_inference.py \
--use_merged_model_checkpoint \
--merged_model_directory /scratch/tathagato/adapter_experiments/extractiveness \
--output_file ./merged_adapter_results/extractiveness_on_extractiveness.json \
--attribute extractiveness

python do_inference.py \
--use_merged_model_checkpoint \
--merged_model_directory /scratch/tathagato/adapter_experiments/topic \
--output_file ./merged_adapter_results/topic_on_topic.json \
--attribute topic

# # Dual attributes
# # length then extractiveness on length
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/length_then_extractiveness \
# --output_file ./merged_adapter_results/length_then_extractiveness_evaluate_length.json \
# --attribute length

# # length then extractiveness on extractiveness
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/length_then_extractiveness \
# --output_file ./merged_adapter_results/length_then_extractiveness_evaluate_extractiveness.json \
# --attribute extractiveness

# # extractiveness then length on length
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/extractiveness_then_length \
# --output_file ./merged_adapter_results/extractiveness_then_length_evaluate_length.json \
# --attribute length

# # extractiveness then length on extractiveness
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/extractiveness_then_length \
# --output_file ./merged_adapter_results/extractiveness_then_length_evaluate_extractiveness.json \
# --attribute extractiveness

# # length then topic on length
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

# # extractiveness then topic on extractiveness
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/extractiveness_then_topic \
# --output_file ./merged_adapter_results/extractiveness_then_topic_evaluate_extractiveness.json \
# --attribute extractiveness

# # extractiveness then topic on topic
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/extractiveness_then_topic \
# --output_file ./merged_adapter_results/extractiveness_then_topic_evaluate_topic.json \
# --attribute topic

# # topic then extractiveness on extractiveness
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/topic_then_extractiveness \
# --output_file ./merged_adapter_results/topic_then_extractiveness_evaluate_extractiveness.json \
# --attribute extractiveness

# # topic then extractiveness on topic
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/topic_then_extractiveness \
# --output_file ./merged_adapter_results/topic_then_extractiveness_evaluate_topic.json \
# --attribute topic


# #single attribute
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/length \
# --output_file  ./merged_adapter_results/length_on_length.json \
# --attribute length

# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/extractiveness \
# --output_file  ./merged_adapter_results/extractiveness_on_extractiveness.json \
# --attribute extractiveness

# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/topic \
# --output_file  ./merged_adapter_results/topic_on_topic.json \
# --attribute topic

# #now dual on both attributes
# #length then extractiveness on length
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/length_then_extractiveness \
# --output_file  ./merged_adapter_results/length_then_extractiveness_evaluate_length.json \
# --attribute length

# #length then extractiveness on extractiveness
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/length_then_extractiveness \
# --output_file  ./merged_adapter_results/length_then_extractiveness_evaluate_extractiveness.json \
# --attribute extractiveness

# #extractiveness then length on length
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/extractiveness_then_length \
# --output_file  ./merged_adapter_results/extractiveness_then_length_evaluate_length.json \
# --attribute length

# #extractiveness then length on extractiveness
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/extractiveness_then_length \
# --output_file  ./merged_adapter_results/extractiveness_then_length_evaluate_extractiveness.json \
# --attribute extractiveness

# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/length_then_topic \
# --output_file  ./merged_adapter_results/length_then_topic_evaluate_length.json \
# --attribute length

# #length then topic on topic
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/length_then_topic \
# --output_file  ./merged_adapter_results/length_then_topic_evaluate_topic.json \
# --attribute topic

# #topic then length on length
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/topic_then_length \
# --output_file  ./merged_adapter_results/topic_then_length_evaluate_length.json \
# --attribute length

# #topic then length on topic
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/topic_then_length \
# --output_file  ./merged_adapter_results/topic_then_length_evaluate_topic.json \
# --attribute topic

# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/extractiveness_then_topic \
# --output_file  ./merged_adapter_results/extractiveness_then_topic_evaluate_extractiveness.json \
# --attribute extractiveness

# #extractiveness then topic on topic
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/extractiveness_then_topic \
# --output_file  ./merged_adapter_results/extractiveness_then_topic_evaluate_topic.json \
# --attribute topic

# #topic then extractiveness on extractiveness
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/topic_then_extractiveness \
# --output_file  ./merged_adapter_results/topic_then_extractiveness_evaluate_extractiveness.json \
# --attribute extractiveness

# #topic then extractiveness on topic
# python do_inference.py \
# --use_merged_model_checkpoint \
# --merged_model_directory /scratch/tathagato/adapter_experiments/topic_then_extractiveness \
# --output_file  ./merged_adapter_results/topic_then_extractiveness_evaluate_topic.json \
# --attribute topic



# #topic then length on topic
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/adapter_experiments/topic \
# --second_checkpoint /scratch/tathagato/adapter_experiments/topic_then_length/length \
# --output_file  ./merged_adapter_results/topic_then_length_evaluate_topic.json \
# --attribute "topic" \
# --first_adapter_name "topic" \
# --second_adapter_name "length"

# #topic then length on length
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/adapter_experiments/topic \
# --second_checkpoint /scratch/tathagato/adapter_experiments/topic_then_length/length \
# --output_file  ./merged_adapter_results/topic_then_length_evaluate_length.json \
# --attribute "length" \
# --first_adapter_name "topic" \
# --second_adapter_name "length"

# #length then topic on topic
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/adapter_experiments/length \
# --second_checkpoint /scratch/tathagato/adapter_experiments/length_then_topic/topic \
# --output_file  ./merged_adapter_results/length_then_topic_evaluate_topic.json \
# --attribute "topic" \
# --first_adapter_name "length" \

# #length then topic on length
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/adapter_experiments/length \
# --second_checkpoint /scratch/tathagato/adapter_experiments/length_then_topic/topic \
# --output_file  ./merged_adapter_results/length_then_topic_evaluate_length.json \
# --attribute "length" \
# --first_adapter_name "length" \
# --second_adapter_name "topic"

# #topic then extractiveness on topic
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/adapter_experiments/topic \
# --second_checkpoint /scratch/tathagato/adapter_experiments/topic_then_extractiveness/extractiveness \
# --output_file  ./merged_adapter_results/topic_then_extractiveness_evaluate_topic.json \
# --attribute "topic" \
# --first_adapter_name "topic" \
# --second_adapter_name "extractiveness"

# #topic then extractiveness on extractiveness
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/adapter_experiments/topic \
# --second_checkpoint /scratch/tathagato/adapter_experiments/topic_then_extractiveness/extractiveness \
# --output_file  ./merged_adapter_results/topic_then_extractiveness_evaluate_extractiveness.json \
# --attribute "extractiveness" \
# --first_adapter_name "topic" \
# --second_adapter_name "extractiveness"

# #extractiveness then topic on topic
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/adapter_experiments/extractiveness \
# --second_checkpoint /scratch/tathagato/adapter_experiments/extractiveness_then_topic/topic \
# --output_file  ./merged_adapter_results/extractiveness_then_topic_evaluate_topic.json \
# --attribute "topic" \
# --first_adapter_name "extractiveness" \
# --second_adapter_name "topic"

# #extractiveness then topic on extractiveness
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/adapter_experiments/extractiveness \
# --second_checkpoint /scratch/tathagato/adapter_experiments/extractiveness_then_topic/topic \
# --output_file  ./merged_adapter_results/extractiveness_then_topic_evaluate_extractiveness.json \   
# --attribute "extractiveness" \
# --first_adapter_name "extractiveness" \
# --second_adapter_name "topic"



# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/adapter_experiments/extractiveness \
# --second_checkpoint /scratch/tathagato/adapter_experiments/length\
# --output_file  ./merged_adapter_results/extractiveness_and_length_evaluate_extractiveness.json \   
# --attribute "extractiveness" \
# --first_adapter_name "extractiveness" \
# --second_adapter_name "length"


# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/adapter_experiments/extractiveness \
# --second_checkpoint /scratch/tathagato/adapter_experiments/length\
# --output_file  ./merged_adapter_results/extractiveness_and_length_evaluate_extractiveness.json \   
# --attribute "extractiveness" \
# --first_adapter_name "extractiveness" \
# --second_adapter_name "length"
  



