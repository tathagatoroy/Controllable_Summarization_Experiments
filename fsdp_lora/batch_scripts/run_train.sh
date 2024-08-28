#!/bin/bash 

cd ../

./batch_scripts/login_huggingface.sh
 
python train.py \
--dataset macsum \
--dataset_samples -1 \
--model_name mistralai/Mistral-7B-Instruct-v0.3 \
--precision bf16  \
--gradient_accumulation_steps 8 \
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
--output_dir /scratch/tathagato/fsdp_qlora_experiments_29_August_mistral/extractiveness \
--lora_rank 32 \
--lora_alpha 16 \
--world_size 4 \
--attribute length \
--log_to wandb \
--lr 5e-5 \
--lr_scheduler cosine \
--apply_gradient_clipping True \
--grad_norm 1.0 \
| tee output1.txt




#inference 



./batch_scripts/login_huggingface.sh


python process_checkpoint.py \
--safetensor_path /scratch/tathagato/fsdp_qlora_experiments_29_August_mistral/extractiveness/model_state_dict_0.safetensors \
--attribute topic \
--dataset_size -1 \
| tee output4.txt

./batch_scripts/login_huggingface.sh


python zero_shot_inference.py \
--attribute extractiveness \
--dataset_size -1 \
--output_dir /scratch/tathagato/fsdp_qlora_experiments_29_August_mistral/zero_shot_inference  | tee output5.txt





