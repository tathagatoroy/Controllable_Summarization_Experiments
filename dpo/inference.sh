#!/bin/bash 

export NCCL_P2P_DISABLE=1 

cd ..
# python do_inference.py --checkpoint_dir /scratch/tathagato/dpo_macsum_storm_llama_extractiveness  --model_id akjindal53244/Llama-3.1-Storm-8B --attribute extractivenes &

# python do_inference.py --checkpoint_dir /scratch/tathagato/dpo_macsum_storm_llama_length  --model_id akjindal53244/Llama-3.1-Storm-8B --attribute length &

# python do_inference.py --checkpoint_dir /scratch/tathagato/dpo_macsum_storm_mistral_extractiveness  --model_id mistralai/Mistral-7B-Instruct-v0.3 --attribute extractivenes &

# python do_inference.py --checkpoint_dir /scratch/tathagato//scratch/tathagato/dpo_macsum_storm_mistral_length  --model_id mistralai/Mistral-7B-Instruct-v0.3 --attribute length &

# Function to run a command and capture its exit status
run_command() {
    local command="$1"
    local job_name="$2"
    eval "$command"
    local exit_status=$?
    if [ $exit_status -eq 0 ]; then
        echo "$job_name: Success"
    else
        echo "$job_name: Failure (Exit code: $exit_status)"
    fi
}

# Run commands on GPU 0
CUDA_VISIBLE_DEVICES=0 python do_inference.py --checkpoint_dir /scratch/tathagato/dpo_macsum_storm_llama_extractiveness --model_id akjindal53244/Llama-3.1-Storm-8B --attribute extractiveness > log1.txt &
pid1=$!

CUDA_VISIBLE_DEVICES=0 python do_inference.py --checkpoint_dir /scratch/tathagato/dpo_macsum_storm_llama_length --model_id akjindal53244/Llama-3.1-Storm-8B --attribute length > log2.txt &
pid2=$!

# Run commands on GPU 1
CUDA_VISIBLE_DEVICES=1 python do_inference.py --checkpoint_dir /scratch/tathagato/dpo_macsum_storm_mistral_extractiveness --model_id mistralai/Mistral-7B-Instruct-v0.3 --attribute extractiveness > log3.txt &
pid3=$!

CUDA_VISIBLE_DEVICES=1 python do_inference.py --checkpoint_dir /scratch/tathagato/dpo_macsum_storm_mistral_length --model_id mistralai/Mistral-7B-Instruct-v0.3 --attribute length > log4.txt &
pid4=$!

# Wait for all processes to finish
wait $pid1 $pid2 $pid3 $pid4

# Check and report status of each job
run_command "wait $pid1" "Llama Extractiveness (GPU 0)"
run_command "wait $pid2" "Llama Length (GPU 0)"
run_command "wait $pid3" "Mistral Extractiveness (GPU 1)"
run_command "wait $pid4" "Mistral Length (GPU 1)"

echo "All processes have completed."