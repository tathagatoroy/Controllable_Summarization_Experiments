#!/bin/bash
#SBATCH -A kcis
#SBATCH -n 55
#SBATCH --gres=gpu:4
#SBATCH --qos=kl4
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=4-00:00:00
#SBATCH --job-name=adapter_fusion_dpo_plus_multi_attribute
#SBATCH --output=/home2/tathagato/summarization/MACSUM/naacl/logs/fusion_dpo_train.out
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

# python run_all_dpo_single_attribute.py
# echo "done first part"
# python run_all_dpo_joint_multi_attribute.py
# echo "done second part"
# python run_all_weighted_adapter_fusion_dpo.py


python run_all_dpo_multi_attribute_single_adapter_continued.py
echo "Completed"
python run_all_dpo_multi_attribute_multi_attribute.py




