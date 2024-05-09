

#first only length trained
python run_inference.py --use_checkpoint \
--model_directory "/scratch/tathagato/openelm_adapter_experiments/2024-05-01-18-26-09_length/checkpoint-4278" \
--attributes "length" 

python run_inference.py --use_checkpoint \
--model_directory "/scratch/tathagato/openelm_adapter_experiments/2024-05-01-18-26-09_length/checkpoint-4278" \
--attributes "extractiveness"

python run_inference.py --use_checkpoint \
--model_directory "/scratch/tathagato/openelm_adapter_experiments/2024-05-01-18-26-09_length/checkpoint-4278" \
--attributes "length" "extractiveness"

#now only extractiveness trained

python run_inference.py --use_checkpoint \
--model_directory "/scratch/tathagato/openelm_adapter_experiments/2024-05-01-19-01-27_extractiveness/checkpoint-8556" \
--attributes "length" 

python run_inference.py --use_checkpoint \
--model_directory "/scratch/tathagato/openelm_adapter_experiments/2024-05-01-19-01-27_extractiveness/checkpoint-8556" \
--attributes "extractiveness"

python run_inference.py --use_checkpoint \
--model_directory "/scratch/tathagato/openelm_adapter_experiments/2024-05-01-19-01-27_extractiveness/checkpoint-8556" \
--attributes "length" "extractiveness"

#length followed by extractiveness

python run_inference.py --use_checkpoint \
--model_directory "/scratch/tathagato/openelm_adapter_experiments/2024-05-02-05-14-47_extractiveness/checkpoint-4278" \
--attributes "length" 

python run_inference.py --use_checkpoint \
--model_directory "/scratch/tathagato/openelm_adapter_experiments/2024-05-02-05-14-47_extractiveness/checkpoint-4278" \
--attributes "extractiveness"

python run_inference.py --use_checkpoint \
--model_directory "/scratch/tathagato/openelm_adapter_experiments/2024-05-02-05-14-47_extractiveness/checkpoint-4278" \
--attributes "length" "extractiveness"

#extractiveness followed by length
python run_inference.py --use_checkpoint \
--model_directory "/scratch/tathagato/openelm_adapter_experiments/2024-05-02-05-14-12_length/checkpoint-4278" \
--attributes "length" 

python run_inference.py --use_checkpoint \
--model_directory "/scratch/tathagato/openelm_adapter_experiments/2024-05-02-05-14-12_length/checkpoint-4278" \
--attributes "extractiveness"

python run_inference.py --use_checkpoint \
--model_directory "/scratch/tathagato/openelm_adapter_experiments/2024-05-02-05-14-12_length/checkpoint-4278" \
--attributes "length" "extractiveness"

