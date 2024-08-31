#!/bin/bash 

cd ../

./batch_scripts/login_huggingface.sh
 
python train_hlora.py \
--dataset macsum \
--dataset_samples 32 \
--model_name mistralai/Mistral-7B-Instruct-v0.3 \
--precision bf16  \
--gradient_accumulation_steps 8 \
--batch_size 1 \
--context_length 2048 \
--num_epochs 1 \
--train_type hlora \
--use_gradient_checkpointing True \
--reentrant_checkpointing True \
--use_cpu_offload True \
--verbose False \
--save_model True \
--verbose True  \
--output_dir /scratch/tathagato/fsdp_qlora_experiments_30_August_mistral_test/extractiveness \
--lora_rank_1 32 \
--lora_rank_2 16 \
--lora_alpha_1 16 \
--lora_alpha_2 8 \
--world_size 4 \
--attribute_1 extractiveness \
--attribute_2 length \
--log_to wandb \
--lr 5e-5 \
--lr_scheduler cosine \
--apply_gradient_clipping True \
--grad_norm 1.0 \
--low_memory True \
| tee output1.txt