#!/bin/bash 

source /home2/tathagato/miniconda3/bin/activate roy

#export only gpu 0
export CUDA_VISIBLE_DEVICES=0

cd ..
#zero shot
# python do_inference.py \
# --output_file  ./results/zero_shot_length.json \
# --attribute length

# python do_inference.py \
# --output_file  ./results/zero_shot_extractiveness.json \
# --attribute extractiveness

# python do_inference.py \
# --output_file  ./results/zero_shot_topic.json \
# --attribute topic

# #single attribute
# python do_inference.py --use_checkpoint \
# --model_directory /scratch/tathagato/new_adapter_experiments/length \
# --output_file  ./results/length_on_length.json \
# --first_adapter_name "length" \
# --attribute length


# python do_inference.py --use_2_checkpoint \
# --first_checkpoint /scratch/tathagato/new_adapter_experiments/extractiveness \
# --second_checkpoint /scratch/tathagato/new_adapter_experiments/length \
# --output_file ./results/extractiveness_and_length_evaluate_extractiveness.json \
# --attribute "extractiveness" \
# --first_adapter_name "extractiveness" \
# --second_adapter_name "length"


#length then topic on topic
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

# #extractiveness then topic on extractiveness
# python do_inference.py --use_2_checkpoint \
# --first_checkpoint_path /scratch/tathagato/new_adapter_experiments/extractiveness \
# --second_checkpoint_path /scratch/tathagato/new_adapter_experiments/extractiveness_then_topic/topic \
# --output_file ./results/extractiveness_then_topic_evaluate_extractiveness.json \
# --attribute "extractiveness" \
# --first_adapter_name "extractiveness" \
# --second_adapter_name "topic"

# extractiveness then topic on extractiveness
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


python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/topic/checkpoint-1/topic \
--attribute topic \
--output_file ./inference_results/topic_evaluate_on_topic_1.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/topic/checkpoint-500/topic \
--attribute topic \
--output_file ./inference_results/topic_evaluate_on_topic_500.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/topic/topic \
--attribute topic \
--output_file ./inference_results/topic_evaluate_on_topic_final.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/specificity/checkpoint-200/specificity \
--attribute specificity \
--output_file ./inference_results/specificity_evaluate_on_specificity_200.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/specificity/checkpoint-400/specificity \
--attribute specificity \
--output_file ./inference_results/specificity_evaluate_on_specificity_400.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi