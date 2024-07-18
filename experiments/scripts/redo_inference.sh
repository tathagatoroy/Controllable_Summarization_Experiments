#!/bin/bash
cd ..


python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/extractiveness/checkpoint-1/extractiveness \
--attribute extractiveness \
--output_file ./inference_results/extractiveness_evaluate_on_extractiveness_1.json  
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/extractiveness/checkpoint-500/extractiveness \
--attribute extractiveness \
--output_file ./inference_results/extractiveness_evaluate_on_extractiveness_500.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/extractiveness/checkpoint-1000/extractiveness \
--attribute extractiveness \
--output_file ./inference_results/extractiveness_evaluate_on_extractiveness_1000.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/extractiveness/checkpoint-1500/extractiveness \
--attribute extractiveness \
--output_file ./inference_results/extractiveness_evaluate_on_extractiveness_1500.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/extractiveness/extractiveness/ \
--attribute extractiveness \
--output_file ./inference_results/extractiveness_evaluate_on_extractiveness_final.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi


python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/length/checkpoint-1/length \
--attribute length \
--output_file ./inference_results/length_evaluate_on_length_1.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/length/checkpoint-500/length \
--attribute length \
--output_file ./inference_results/length_evaluate_on_length_500.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/length/checkpoint-1000/length \
--attribute length \
--output_file ./inference_results/length_evaluate_on_length_1000.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/length/checkpoint-1500/length \
--attribute length \
--output_file ./inference_results/length_evaluate_on_length_1500.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/length/length/ \
--attribute length \
--output_file ./inference_results/length_evaluate_on_length_final.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/topic/checkpoint-1/topic \
--attribute topic \
--output_file ./inference_results/topic_evaluate_on_topic_1.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/topic/checkpoint-500/topic \
--attribute topic \
--output_file ./inference_results/topic_evaluate_on_topic_500.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi

python do_inference.py \
--use_checkpoint \
--checkpoint_path /scratch/tathagato/redo_adapter_experiments/topic/topic \
--attribute topic \
--output_file ./inference_results/topic_evaluate_on_topic_final.json
if [[ $? = 0 ]]; then
        echo "success"
else
        echo "failure"
fi
