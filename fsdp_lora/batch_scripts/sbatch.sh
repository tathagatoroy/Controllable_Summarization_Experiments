#!/bin/bash

#SBATCH -A research
#SBATCH -n 38
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=4-00:00:00
#SBATCH --job-name=fsdp_lora_Finetune
#SBATCH --output=finetune.out
#SBATCH --mail-user=tathagato.roy@research.iiit.ac.in
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH -w gnode051

./login_huggingface.sh

# Remove the temporary file after login
rm -f "$TEMP_FILE"
./run_train.sh

echo "Completed"




