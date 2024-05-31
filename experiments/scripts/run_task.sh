#!/bin/bash
cd ..


#accelerate launch test_finetune_phi.py \
# --attribute length \
#--output_dir /scratch/tathagato/adapter_experiments/length

#accelerate launch test_finetune_phi.py \
# --attribute extractiveness \
#--output_dir /scratch/tathagato/adapter_experiments/extractiveness

#accelerate launch test_finetune_phi.py \
# --attribute topic \
#--output_dir /scratch/tathagato/adapter_experiments/topic

#length then topic
accelerate launch test_finetune_phi.py \
 --attribute topic \
--load_previous_model \
--previous_model_path /scratch/tathagato/adapter_experiments/length/length \
--output_dir /scratch/tathagato/adapter_experiments/length_then_topic

#length then extractiveness
accelerate launch test_finetune_phi.py \
 --attribute extractiveness \
--load_previous_model \
--previous_model_path /scratch/tathagato/adapter_experiments/length/length \
--output_dir /scratch/tathagato/adapter_experiments/length_then_extractiveness




#extractiveness then topic
accelerate launch test_finetune_phi.py \
 --attribute topic \
--load_previous_model \
--previous_model_path /scratch/tathagato/adapter_experiments/extractiveness/extractiveness \
--output_dir /scratch/tathagato/adapter_experiments/extractiveness_then_topic


#extractiveness then length
accelerate launch test_finetune_phi.py \
 --attribute length \
--load_previous_model \
--previous_model_path /scratch/tathagato/adapter_experiments/extractiveness/extractiveness \
--output_dir /scratch/tathagato/adapter_experiments/extractiveness_then_length

#topic then length
accelerate launch test_finetune_phi.py \
 --attribute length \
--load_previous_model \
--previous_model_path /scratch/tathagato/adapter_experiments/topic/topic \
--output_dir /scratch/tathagato/adapter_experiments/topic_then_length

#topic then extractiveness
accelerate launch test_finetune_phi.py \
 --attribute extractiveness \
--load_previous_model \
--previous_model_path /scratch/tathagato/adapter_experiments/topic/topic \
--output_dir /scratch/tathagato/adapter_experiments/topic_then_extractiveness






