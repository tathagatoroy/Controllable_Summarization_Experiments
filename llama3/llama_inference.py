import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import sys
import os
import tqdm
import pickle as pkl

# Import custom dataset
sys.path.append("/home2/tathagato/summarization/MACSUM/fsdp_qlora")
from dataset import MACSUM

def main(args):
    # Set up tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Set up the dataset
    dataset = MACSUM(tokenizer=tokenizer, attribute=args.attribute, size=args.dataset_size)
    print(f"Dataset size: {len(dataset)}")

    # Set up quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map
    )

    # Prepare the results directory
    directory = os.path.dirname(args.result_path)
    os.makedirs(directory, exist_ok=True)

    # Process dataset and generate summaries
    result_dict = {}
    for index, item in tqdm.tqdm(enumerate(dataset), desc="Generating summaries"):
        new_item = {key: value.to('cuda') for key, value in item.items() if key != 'reference'}
        output = model.generate(**new_item, do_sample=args.do_sample, top_p=args.top_p, top_k=args.top_k, max_new_tokens=args.max_new_tokens, num_return_sequences=args.num_return_sequences)
        decoded_text = tokenizer.decode(output[0], skip_special_tokens=True)
        just_summary = decoded_text.split("Response:")[-1].strip("\n")
        #print(just_summary)
        #print("------------------")
        #print(decoded_text)
        #print("====================================")
        input_text = decoded_text.split("Response:")[0]
        result_dict[index] = {'input': input_text, 'summary': just_summary, 'reference': item['reference'], 'generated_text' : decoded_text}
    # Save the result_dict
    with open(args.result_path, 'wb') as f:
        pkl.dump(result_dict, f)

    # Optional: interactive debugging session
    #if args.debug:
    #import code; code.interact(local=locals())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for quantized LLaMA model inference.")

    parser.add_argument('--model_id', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model ID for the transformer model.")
    parser.add_argument('--attribute', type=str, default="length", help="Attribute used for filtering the dataset.")
    parser.add_argument('--dataset_size', type=int, default=3, help="Size of the dataset to be used.")
    parser.add_argument('--result_path', type=str, default="/scratch/tathagato/llama_results/llama3.1_8b_zero_shot_length.pkl", help="Path to save the result dictionary.")
    parser.add_argument('--load_in_4bit', type=bool, default=True, help="Whether to load the model in 4-bit mode.")
    parser.add_argument('--bnb_4bit_quant_type', type=str, default="nf4", help="Quantization type for BitsAndBytes.")
    parser.add_argument('--bnb_4bit_compute_dtype', type=torch.dtype, default=torch.float16, help="Compute dtype for 4-bit quantization.")
    parser.add_argument('--bnb_4bit_use_double_quant', type=bool, default=True, help="Whether to use double quantization.")
    parser.add_argument('--torch_dtype', type=torch.dtype, default=torch.float16, help="Torch data type for the model.")
    parser.add_argument('--device_map', type=str, default='auto', help="Device map for model placement.")
    parser.add_argument('--do_sample', type=bool, default=True, help="Whether to sample during text generation.")
    parser.add_argument('--top_p', type=float, default=0.95, help="Top-p for nucleus sampling.")
    parser.add_argument('--top_k', type=int, default=50, help="Top-k for top-k sampling.")
    parser.add_argument('--max_new_tokens', type=int, default=300, help="Maximum number of new tokens to generate.")
    parser.add_argument('--num_return_sequences', type=int, default=1, help="Number of sequences to return during generation.")
    parser.add_argument('--debug', action='store_true', help="Enable interactive debugging session after script execution.")

    args = parser.parse_args()
    main(args)
