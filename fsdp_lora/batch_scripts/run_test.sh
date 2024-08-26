#!/bin/bash 

cd ../

./batch_scripts/login_huggingface.sh
 
python train.py \
--dataset macsum \
--dataset_samples 32 \
--model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
--precision bf16  \
--gradient_accumulation_steps 4 \
--batch_size 1 \
--context_length 2048 \
--num_epochs 3 \
--train_type qlora \
--use_gradient_checkpointing True \
--reentrant_checkpointing True \
--use_cpu_offload True \
--verbose False \
--save_model True \
--verbose True  \
--output_dir /scratch/tathagato/fsdp_qlora_experiments_25_August_test/length \
--lora_rank 16 \
--world_size 4 \
--attribute length \
--log_to wandb \
--lr 5e-4 \
--lr_scheduler cosine \
| tee output1.txt


./batch_scripts/login_huggingface.sh

python train.py \
--dataset macsum \
--dataset_samples 32 \
--model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
--precision bf16  \
--gradient_accumulation_steps 4 \
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
--output_dir /scratch/tathagato/fsdp_qlora_experiments_25_August_test/extractiveness \
--lora_rank 16 \
--world_size 4 \
--attribute extractiveness \
--log_to wandb \
--lr 5e-4 \
--lr_scheduler cosine \
| tee output2.txt