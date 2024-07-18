#!/bin/bash
cd /home2/tathagato/summarization/MACSum/experiments


accelerate launch test_finetune_phi.py \
--attribute length \
--learning_rate 1e-5 \
--output_dir /scratch/tathagato/non_packed_adapter_experiments/length

accelerate launch test_finetune_phi.py \
--attribute extractiveness \
--learning_rate 1e-5 \
--output_dir /scratch/tathagato/non_packed_adapter_experiments/extractiveness

accelerate launch test_finetune_phi.py \
--attribute topic \
--learning_rate 1e-5 \
--output_dir /scratch/tathagato/non_packed_adapter_experiments/topic


accelerate launch test_finetune_phi.py \
--attribute specificity \
--learning_rate 1e-5 \
--output_dir /scratch/tathagato/non_packed_adapter_experiments/specificity


accelerate launch test_finetune_phi.py \
--attribute length \
--learning_rate 6e-4 \
--output_dir /scratch/tathagato/non_packed_adapter_experiments/length

accelerate launch test_finetune_phi.py \
--attribute extractiveness \
--learning_rate 6e-4 \
--output_dir /scratch/tathagato/non_packed_adapter_experiments/extractiveness

accelerate launch test_finetune_phi.py \
--attribute topic \
--learning_rate 6e-4 \
--output_dir /scratch/tathagato/non_packed_adapter_experiments/topic


accelerate launch test_finetune_phi.py \
--attribute specificity \
--learning_rate 6e-4 \
--output_dir /scratch/tathagato/non_packed_adapter_experiments/specificity


accelerate launch test_finetune_phi.py \
--attribute length \
--learning_rate 1e-4 \
--output_dir /scratch/tathagato/non_packed_adapter_experiments/length

accelerate launch test_finetune_phi.py \
--attribute extractiveness \
--learning_rate 1e-4 \
--output_dir /scratch/tathagato/non_packed_adapter_experiments/extractiveness

accelerate launch test_finetune_phi.py \
--attribute topic \
--learning_rate 1e-4 \
--output_dir /scratch/tathagato/non_packed_adapter_experiments/topic


accelerate launch test_finetune_phi.py \
--attribute specificity \
--learning_rate 1e-4 \
--output_dir /scratch/tathagato/non_packed_adapter_experiments/specificity

# accelerate launch test_finetune_phi.py \
# --attribute Speaker 
#--output_dir /scratch/tathagato/non_packed_adapter_experiments/topic

#length then topic
# accelerate launch test_finetune_phi.py \
#  --attribute topic \
# --load_previous_model \
# --previous_model_path /scratch/tathagato/non_packed_adapter_experiments/length/length \

# #length then extractiveness
# accelerate launch test_finetune_phi.py \
#  --attribute extractiveness \
# --load_previous_model \
# --previous_model_path /scratch/tathagato/non_packed_adapter_experiments/length/length \




# #extractiveness then topic
# accelerate launch test_finetune_phi.py \
#  --attribute topic \
# --load_previous_model \
# --previous_model_path /scratch/tathagato/redo_adapter_experiments/extractiveness/extractiveness \
# # --output_dir /scratch/tathagato/redo_adapter_experiments/extractiveness_then_topic


# #extractiveness then length
# accelerate launch test_finetune_phi.py \
#  --attribute length \
# --load_previous_model \
# --previous_model_path /scratch/tathagato/redo_adapter_experiments/extractiveness/extractiveness \
# # --output_dir /scratch/tathagato/redo_adapter_experiments/extractiveness_then_length

# #topic then length
# accelerate launch test_finetune_phi.py \
#  --attribute length \
# --load_previous_model \
# --previous_model_path /scratch/tathagato/redo_adapter_experiments/topic/topic \
# # --output_dir /scratch/tathagato/redo_adapter_experiments/topic_then_length

# #topic then extractiveness
# accelerate launch test_finetune_phi.py \
#  --attribute extractiveness \
# --load_previous_model \
# --previous_model_path /scratch/tathagato/redo_adapter_experiments/topic/topic \
# --output_dir /scratch/tathagato/redo_adapter_experiments/topic_then_extractiveness






