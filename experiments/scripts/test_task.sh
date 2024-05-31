#!/bin/bash


cd ..
#zero shot
python do_inference.py \
--output_file  ./results/zero_shot_length \
--attribute length

# python do_inference.py \
# --output_file  ./results/zero_shot_length \
# --attribute length

# python do_inference.py \
# --output_file  ./results/zero_shot_length \
# --attribute length

#single attribute
python do_inference.py --use_checkpoint \
--model_directory /scratch/tathagato/new_adapter_experiments/length \
--output_file  ./results/length_on_length.json \
--first_adapter_name "length" \
--attribute length

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

#now dual on both attributes
#length then extractiveness on length
python do_inference.py --use_2_checkpoint \
--first_checkpoint /scratch/tathagato/new_adapter_experiments/length \
--second_checkpoint /scratch/tathagato/new_adapter_experiments/length_then_extractiveness/extractiveness \
--output_file  ./results/length_then_extractiveness_evaluate_length.json \
--attribute length
--first_adapter_name "length" \
--second_adapter_name "extractiveness"

#length then extractiveness on extractiveness
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/new_adapter_experiments/length \
# --second_checkpoint /scratch/tathagato/new_adapter_experiments/length_then_extractiveness/extractiveness \
# --output_file  ./results/length_then_extractiveness_evaluate_extractiveness.json \
# --attribute "extractiveness"
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

# #length then topic on topic
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/new_adapter_experiments/length \
# --second_checkpoint /scratch/tathagato/new_adapter_experiments/length_then_topic/topic \
# --output_file  ./results/length_then_topic_evaluate_topic.json \
# --attribute "topic" \
# --first_adapter_name "length" \

# #length then topic on length
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/new_adapter_experiments/length \
# --second_checkpoint /scratch/tathagato/new_adapter_experiments/length_then_topic/topic \
# --output_file  ./results/length_then_topic_evaluate_length.json \
# --attribute "length" \
# --first_adapter_name "length" \
# --second_adapter_name "topic"

# #topic then extractiveness on topic
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

# #extractiveness then topic on extractiveness
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/new_adapter_experiments/extractiveness \
# --second_checkpoint /scratch/tathagato/new_adapter_experiments/extractiveness_then_topic/topic \
# --output_file  ./results/extractiveness_then_topic_evaluate_extractiveness.json \   
# --attribute "extractiveness" \
# --first_adapter_name "extractiveness" \
# --second_adapter_name "topic"
  



