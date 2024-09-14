#!/bin/bash

cd ../

# Run the first training script on GPU 0
CUDA_VISIBLE_DEVICES=0 python train_hlora.py --output_dir /scratch/tathagato/length_then_extractiveness --attribute_1 length --attribute_2 extractiveness --model_id akjindal53244/Llama-3.1-Storm-8B > output_1.txt &
pid1 = $!

# Run the second training script on GPU 1
CUDA_VISIBLE_DEVICES=1 python train_hlora.py --output_dir /scratch/tathagato/extractiveness_then_length --attribute_2 length --attribute_1 extractiveness --model_id akjindal53244/Llama-3.1-Storm-8B > output_2.txt &
pid2 = $! 

# Run the first training script on GPU 0
CUDA_VISIBLE_DEVICES=0 python train_hlora.py --output_dir /scratch/tathagato/length_then_extractiveness --attribute_1 length --attribute_2 extractiveness --model_id mistralai/Mistral-7B-Instruct-v0.3 > output_3.txt &
pid1 = $!

# Run the second training script on GPU 1
CUDA_VISIBLE_DEVICES=1 python train_hlora.py --output_dir /scratch/tathagato/extractiveness_then_length --attribute_2 length --attribute_1 extractiveness --model_id mistralai/Mistral-7B-Instruct-v0.3 > output_24.txt &
pid2 = $! 

wait $pid1 $pid2
echo "done"

