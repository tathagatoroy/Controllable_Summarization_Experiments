#!/bin/bash 

cd /home2/tathagato/summarization/MACSUM/fsdp_lora/batch_scripts/
pwd 
./login_huggingface.sh
cd /home2/tathagato/summarization/MACSUM/dpo/
pwd 

export TOKENIZERS_PARALLELISM=False
WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1234 train_dpo.py \
    --model_name_or_path=mistralai/Mistral-7B-Instruct-v0.3 \
    --per_device_train_batch_size 12 \
    --ddp_find_unused_parameters=False \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --output_dir="/scratch/tathagato/dpo_macsum_storm_mistral_length" \
    --optim adamw_torch_fused \
    --warmup_steps 30 \
    --report_to wandb \
    --bf16 \
    --torch_dtype bfloat16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=32 \
    --sanity_check \
    --lora_target_modules all \
    --attn_implementation flash_attention_2 \
    --load_in_4bit \
    --use_bnb_nested_quant \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --report_to wandb \
    --attributes "length" \
    --lora_alpha=16 > ./logs/output_mistral_length.txt

cd /home2/tathagato/summarization/MACSUM/fsdp_lora/batch_scripts/
pwd 
./login_huggingface.sh
cd /home2/tathagato/summarization/MACSUM/dpo/
pwd 

WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1234 train_dpo.py \
    --model_name_or_path=akjindal53244/Llama-3.1-Storm-8B \
    --per_device_train_batch_size 4 \
    --ddp_find_unused_parameters=False \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --output_dir="/scratch/tathagato/dpo_macsum_storm_llama_length" \
    --optim adamw_torch_fused \
    --warmup_steps 30 \
    --report_to wandb \
    --bf16 \
    --torch_dtype bfloat16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=32 \
    --sanity_check \
    --lora_target_modules all \
    --attn_implementation flash_attention_2 \
    --load_in_4bit \
    --use_bnb_nested_quant \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --report_to wandb \
    --attributes "length" \
    --lora_alpha=16 > ./logs/output_llama_length.txt

cd /home2/tathagato/summarization/MACSUM/fsdp_lora/batch_scripts/
pwd 
./login_huggingface.sh
cd /home2/tathagato/summarization/MACSUM/dpo/
pwd 


export TOKENIZERS_PARALLELISM=False
WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1234 train_dpo.py \
    --model_name_or_path=mistralai/Mistral-7B-Instruct-v0.3 \
    --per_device_train_batch_size 12 \
    --ddp_find_unused_parameters=False \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --output_dir="/scratch/tathagato/dpo_macsum_storm_mistral_extractiveness" \
    --optim adamw_torch_fused \
    --warmup_steps 30 \
    --report_to wandb \
    --bf16 \
    --torch_dtype bfloat16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=32 \
    --sanity_check \
    --lora_target_modules all \
    --attn_implementation flash_attention_2 \
    --load_in_4bit \
    --use_bnb_nested_quant \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --report_to wandb \
    --attributes "extractiveness" \
    --lora_alpha=16 > ./logs/output_mistral_extractiveness.txt

cd /home2/tathagato/summarization/MACSUM/fsdp_lora/batch_scripts/
pwd 
./login_huggingface.sh
cd /home2/tathagato/summarization/MACSUM/dpo/
pwd 

WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1234 train_dpo.py \
    --model_name_or_path=akjindal53244/Llama-3.1-Storm-8B \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --ddp_find_unused_parameters=False \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --output_dir="/scratch/tathagato/dpo_macsum_storm_llama_extractiveness" \
    --optim adamw_torch_fused \
    --warmup_steps 30 \
    --report_to wandb \
    --bf16 \
    --torch_dtype bfloat16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=32 \
    --sanity_check \
    --lora_target_modules all \
    --attn_implementation flash_attention_2 \
    --attributes "extractiveness" \
    --load_in_4bit \
    --use_bnb_nested_quant \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --report_to wandb \
    --lora_alpha=16 > ./logs/output_llama_extractiveness.txt



    
