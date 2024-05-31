#!/bin/bash

#SBATCH -A research
#SBATCH -n 35
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=4-00:00:00
#SBATCH --job-name=one_attribute_inference
#SBATCH --output=inference.out
#SBATCH --mail-user=tathagato.roy@research.iiit.ac.in
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH -w gnode090




echo "Activating Conda Environment Virtual Environment"
source /home2/tathagato/miniconda3/bin/activate roy
echo "running script"
./run_inference.sh
# accelerate launch test_finetune_phi.py --attribute length
# accelerate launch test_finetune_phi.py --attribute extractiveness
# accelerate launch test_finetune_phi.py --attribute topic

# accelerate launch test_finetune_phi.py \
#  --attribute length \
# --output_dir /scratch/tathagato/new_adapter_experiments/length
# echo "length done"

# accelerate launch test_finetune_phi.py \
#  --attribute topic \
# --output_dir /scratch/tathagato/new_adapter_experiments/topic
# echo "topic done"

# accelerate launch test_finetune_phi.py \
#  --attribute extractiveness \
# --output_dir /scratch/tathagato/new_adapter_experiments/extractiveness
# echo "extractiveness done"

# #extractiveness and topic
# accelerate launch test_finetune_phi.py \
#  --attribute topic \
# --load_previous_model \
# --previous_model_path /scratch/tathagato/new_adapter_experiments/extractiveness \
# --output_dir /scratch/tathagato/new_adapter_experiments/extractiveness_then_topic
# echo "extractiveness and topic done"
# #length and topic
# accelerate launch test_finetune_phi.py \
#  --attribute topic \
# --load_previous_model \
# --previous_model_path /scratch/tathagato/new_adapter_experiments/length \
# --output_dir /scratch/tathagato/new_adapter_experiments/length_then_topic
# echo "length and topic done"

# #length and extractiveness
# accelerate launch test_finetune_phi.py \
#  --attribute extractiveness \
# --load_previous_model \
# --previous_model_path /scratch/tathagato/new_adapter_experiments/length \
# --output_dir /scratch/tathagato/new_adapter_experiments/length_then_extractiveness
# echo "length and extractiveness done"

# #extractiveness and length
# accelerate launch test_finetune_phi.py \
#  --attribute length \
# --load_previous_model \
# --previous_model_path /scratch/tathagato/new_adapter_experiments/extractiveness \
# --output_dir /scratch/tathagato/new_adapter_experiments/extractiveness_then_length
# echo "extractiveness and length done"

# #topic and length
# accelerate launch test_finetune_phi.py \
#  --attribute length \
# --load_previous_model \
# --previous_model_path /scratch/tathagato/new_adapter_experiments/topic \
# --output_dir /scratch/tathagato/new_adapter_experiments/topic_then_length
# echo "topic and length done"

# #topic and extractiveness
# accelerate launch test_finetune_phi.py \
#  --attribute extractiveness \
# --load_previous_model \
# --previous_model_path /scratch/tathagato/new_adapter_experiments/topic \
# --output_dir /scratch/tathagato/new_adapter_experiments/topic_then_extractiveness
# echo "topic and extractiveness done"

echo "Completed"




