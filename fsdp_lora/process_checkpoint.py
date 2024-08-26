from safetensors import safe_open
import torch
from transformers import LlamaForCausalLM, BitsAndBytesConfig, AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import os 
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys
import pickle as pkl
from dataset import MACSUM
import tqdm
import time


def filter_and_save_memory(tensors):
    for key in tensors.keys():
        if "lora" not in key:
            tensors[key] = None
    return tensors
        

def load_safetensors(safetensor_path):
    tensors = {}
    with safe_open(safetensor_path, framework="pt", device=0) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors


def update_state_dict(state_dict, tensors):
    for key in tensors.keys():
        if "lora" in key and key in state_dict:
            state_dict[key] = tensors[key]
    return state_dict


def return_peft_model(model_id, rank = 32, alpha = 16):
    
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_cache=False,
    quantization_config=bnb_config,
    device_map="cuda:0")

    
    # Add LoRA (make sure your rank (r) and alpha (lora_alpha) values match those used in training!)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=rank, lora_alpha=alpha, lora_dropout=0.1,
        target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
        )
    model = get_peft_model(model, peft_config)
    return model 


def cast_lora_to_bf16(model):
    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16) 
        if param.dtype == torch.float16:
            param.data = param.data.to(torch.bfloat16)
    
    #print everything 
    for name, param in model.named_parameters():
        print(f"name : {name} | dtype : {param.dtype} | shape : {param.shape} | requires_grad : {param.requires_grad} \n\n")

    return model
def load_saved_model_and_save_model(safetensor_path, model_id):
    tensors = load_safetensors(safetensor_path)
    tensors = filter_and_save_memory(tensors)
    model = return_peft_model(model_id)
    state_dict = model.state_dict()
    state_dict = update_state_dict(state_dict, tensors)
    model.load_state_dict(state_dict, strict=False)
    model = cast_lora_to_bf16(model)
    dirname = os.path.dirname(safetensor_path)
    model.save_pretrained(dirname)
    return model



    #covert




    







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--safetensor_path", type=str, default = "/scratch/tathagato/fsdp_qlora_experiments_25_August_test_llama3.1/length/model_state_dict_0.safetensors" )
    parser.add_argument("--model_id", type=str, default= "mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--attribute", type=str, default="length")
    parser.add_argument("--dataset_size", type=int, default=3)
    parser.add_argument('--do_sample', type=bool, default=True, help="Whether to sample during text generation.")
    parser.add_argument('--top_p', type=float, default=0.95, help="Top-p for nucleus sampling.")
    parser.add_argument('--top_k', type=int, default=50, help="Top-k for top-k sampling.")
    parser.add_argument('--max_new_tokens', type=int, default=300, help="Maximum number of new tokens to generate.")
    parser.add_argument('--num_return_sequences', type=int, default=1, help="Number of sequences to return during generation.")

    args = parser.parse_args()
    model = load_saved_model_and_save_model(args.safetensor_path, args.model_id)

    #test with tensors
    tensors = load_safetensors(args.safetensor_path)
    tensors = filter_and_save_memory(tensors)
    for key in tensors.keys():
        if tensors[key] is not None and key in model.state_dict():
            model_state = model.state_dict()[key]
            assert torch.allclose(model_state, tensors[key], atol=1e-3)
    print("All tensors are equal")
    model_type = "llama3.1"
    if "mistral" in args.model_id:
        model_type = "mistral"
    print("---------------------------------------------------")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    dataset = MACSUM(tokenizer=tokenizer, attribute=args.attribute, size=args.dataset_size, mode = "inference", model_type= model_type)
    print(f"Dataset size : {len(dataset)}")

    print("example of the dataset")
    input_ids = dataset[0]["input_ids"]
    print(f"input_ids : {input_ids}")
    print(f"prompt : {tokenizer.decode(input_ids[0])}")



    reference = dataset[0]["reference"]
    print(f"reference : {reference}")


    print("---------------------------------------------------")
    # Prepare the results directory
    directory = os.path.dirname(args.safetensor_path)
    os.makedirs(directory, exist_ok=True)
    result_path = os.path.join(directory, "result.pkl")

    # Process dataset and generate summaries
    result_dict = {}
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for index, item in tqdm.tqdm(enumerate(dataset), desc="Generating summaries"):
            new_item = {key: value.to('cuda') for key, value in item.items() if key != 'reference'}
            output = model.generate(**new_item, do_sample=args.do_sample, top_p=args.top_p, top_k=args.top_k, max_new_tokens=args.max_new_tokens, num_return_sequences=args.num_return_sequences)
            
            decoded_text = tokenizer.decode(output[0], skip_special_tokens=True)
            generation_ids = output[0,len(new_item["input_ids"][0]):]
            generated_text = tokenizer.decode(generation_ids, skip_special_tokens=True)
            prompt = tokenizer.decode(new_item["input_ids"][0], skip_special_tokens=True)
            result_dict[index] = {'input': prompt, 'summary': generated_text, 'reference': item['reference'], 'generated_text' : decoded_text}
        end_time = time.time()
    print(f"Total time taken for generating summaries : {end_time - start_time}")
    print(f"Average time taken for generating summaries : {(end_time - start_time)/len(dataset)}")
    # Save the result_dict
    with open(result_path, 'wb') as f:
        pkl.dump(result_dict, f)

    for key in result_dict.keys():
        print(f"input : {result_dict[key]['input']}")
        print(f"summary : {result_dict[key]['summary']}")
        print(f"reference : {result_dict[key]['reference']}")
        print(f"generated_text : {result_dict[key]['generated_text']}")
        print("---------------------------------------------------")




    

