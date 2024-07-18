import os
import subprocess
from concurrent.futures import ThreadPoolExecutor


checkpoint_dir = "/scratch/tathagato/non_packed_adapter_experiments"
output_dir = "/scratch/tathagato/non_packed_experiment_outputs"
log_dir = "/scratch/tathagato/logs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)




# Number of GPUs available
num_gpus = 4

def run_command_on_gpu(gpu_id, process_num, output_file, dir, attribute, index):
    command = f"python ./../do_inference.py --use_checkpoint --checkpoint_path {dir} --attribute {attribute} --output_file {output_dir}/{output_file}.json"
    print(f"running experiment {index} \n command : {command} \n log file gpu_{gpu_id}_process_{index}.log \n\n")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_file = os.path.join(log_dir, f"gpu_{gpu_id}_process_{index}.log")
    with open(log_file, "w") as log:
        result = subprocess.run(command, shell=True, env=env, stdout=log, stderr=log)
    if result.returncode == 0:
        print(f"Process {process_num} on GPU {gpu_id} succeeded")
    else:
        print(f"Process {process_num} on GPU {gpu_id} failed with return code {result.returncode}")

# Function to run commands in sequence on a given GPU
def process_commands_for_gpu(gpu_id, commands):
    for process_num, command in enumerate(commands):
        run_command_on_gpu(gpu_id, process_num, *command)

def find_safetensors_dirs(base_path):
    safetensors_dirs = set()

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".safetensors"):
                safetensors_dirs.add(root)
                break  # No need to check more files in this directory

    return safetensors_dirs

safetensors_dirs = find_safetensors_dirs(checkpoint_dir)
output_files = [x.replace(checkpoint_dir,"").replace("/","_")[1:] for x in safetensors_dirs]
attributes = [os.path.basename(x) for x in safetensors_dirs]

# num_examples_to_process = 1
# safetensors_dirs = list(safetensors_dirs)[:num_examples_to_process]
# output_files = output_files[:num_examples_to_process]
# attributes = attributes[:num_examples_to_process]
print("number of experiments: ", len(safetensors_dirs))


# Group commands by GPU
commands_per_gpu = [[] for _ in range(num_gpus)]
for i, (output_file, dir, attribute) in enumerate(zip(output_files, safetensors_dirs, attributes)):
    gpu_id = i % num_gpus  # Cycle through the GPUs
    commands_per_gpu[gpu_id].append((output_file, dir, attribute, i))

# Execute commands for each GPU sequentially in parallel
with ThreadPoolExecutor(max_workers=num_gpus) as executor:
    futures = []
    for gpu_id in range(num_gpus):
        future = executor.submit(process_commands_for_gpu, gpu_id, commands_per_gpu[gpu_id])
        futures.append(future)

    for future in futures:
        future.result()  # Wait for all futures to complete
