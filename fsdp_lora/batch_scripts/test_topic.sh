#!/bin/bash 

cd ../




./batch_scripts/login_huggingface.sh

python train.py \
--dataset macsum \
--dataset_samples -1 \
--model_name mistralai/Mistral-7B-Instruct-v0.3 \
--precision bf16  \
--gradient_accumulation_steps 2 \
--batch_size 1 \
--context_length 2048 \
--num_epochs 1 \
--train_type qlora \
--use_gradient_checkpointing True \
--reentrant_checkpointing True \
--use_cpu_offload True \
--verbose False \
--save_model True \
--verbose True  \
--output_dir /scratch/tathagato/fsdp_qlora_experiments_25_August_test_topic/topic \
--lora_rank 32 \
--lora_alpha 16 \
--world_size 4 \
--attribute topic \
--log_to wandb \
--lr 1e-6 \
--lr_scheduler cosine \
--apply_gradient_clipping True \
--grad_norm 1.0 | tee output2.txt