#!/bin/bash
#SBATCH -A kcis
#SBATCH -n 55
#SBATCH --gres=gpu:4
#SBATCH --qos=kl4
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=4-00:00:00
#SBATCH --job-name=adapter_fusion
#SBATCH --output=/home2/tathagato/summarization/MACSUM/naacl/logs/fusion.out
#SBATCH --partition=lovelace
#SBATCH --mail-user=tathagato.roy@research.iiit.ac.in
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH -N 1
#SBATCH -w gnode121
export NCCL_P2P_DISABLE=1

python run_all_dpo_single_attribute.py

echo "Completed"




