#!/bin/bash

#SBATCH -A research
#SBATCH -n 38
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=4-00:00:00
#SBATCH --job-name=69_sdp_lora_Finetune_inference
#SBATCH --output=finetune.out
#SBATCH --mail-user=tathagato.roy@research.iiit.ac.in
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH -w gnode069

./login_huggingface.sh

# Remove the temporary file after login
rm -f "$TEMP_FILE"
./run_train.sh

echo "Completed"




