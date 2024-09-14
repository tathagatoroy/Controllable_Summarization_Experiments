import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

def run_command(command, output_file):
    try:
        with open(output_file, 'w') as f:
            process = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, text=True, shell=True)
            process.wait()
            if process.returncode == 0:
                print(f"Success: {command}")
            else:
                print(f"Error: {command} failed with return code {process.returncode}")
    except Exception as e:
        print(f"Error: {command} failed with exception: {str(e)}")

def run_gpu_0_tasks():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    commands = [
        "CUDA_VISIBLE_DEVICES=0 python train_hlora.py --output_dir /scratch/tathagato/mistral_length_then_extractiveness --attribute_1 length --attribute_2 extractiveness --model_id mistralai/Mistral-7B-Instruct-v0.3 ",
        "CUDA_VISIBLE_DEVICES=0 python train_hlora.py --output_dir /scratch/tathagato/mistral_extractiveness_then_length --attribute_2 length --attribute_1 extractiveness --model_id mistralai/Mistral-7B-Instruct-v0.3 "
    ]
    for i, command in enumerate(commands, start=1):
        run_command(command, f"mistral_output_{i}.txt")

def run_gpu_1_tasks():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    commands = [
        "CUDA_VISIBLE_DEVICES=1 python train_hlora.py --output_dir /scratch/tathagato/storm_llama31_length_then_extractiveness --attribute_1 length --attribute_2 extractiveness --model_id akjindal53244/Llama-3.1-Storm-8B ",
        "CUDA_VISIBLE_DEVICES=1 python train_hlora.py --output_dir /scratch/tathagato/storm_llama31_extractiveness_then_length --attribute_2 length --attribute_1 extractiveness --model_id akjindal53244/Llama-3.1-Storm-8B "
    ]
    for i, command in enumerate(commands, start=3):
        run_command(command, f"llama_output_{i}.txt")

if __name__ == "__main__":
    os.chdir("..")  # Change to parent directory
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        gpu_0_future = executor.submit(run_gpu_0_tasks)
        gpu_1_future = executor.submit(run_gpu_1_tasks)
        
        gpu_0_future.result()
        gpu_1_future.result()
    
    print("All tasks completed.")