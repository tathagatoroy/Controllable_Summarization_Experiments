#!/bin/bash
#SBATCH -A kcis
#SBATCH -n 27
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=4-00:00:00
#SBATCH --job-name=dpo
#SBATCH --output=finetune.out
#SBATCH --partition=lovelace
#SBATCH --mail-user=tathagato.roy@research.iiit.ac.in
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH -w gnode121
export NCCL_P2P_DISABLE=1

# ./login_huggingface.sh

# # Remove the temporary file after login
# rm -f "$TEMP_FILE"
# ./test_dpo.sh
./inference.sh

echo "Completed"




