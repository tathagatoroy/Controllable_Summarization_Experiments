import sys
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit
import datasets
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, set_seed
import torch.nn as nn
import tqdm
from datasets import load_dataset
from peft import LoraConfig,PeftConfig, PeftModel, PeftModelForCausalLM
import torch
import torch.nn.functional as F
import argparse
from datetime import datetime
import os
from utils import load_dataset_from_path_phi, load_multi_attribute_dataset_from_path,print_trainable_parameters, count_model_parameters
from transformers import pipeline
import json
from dataclasses import dataclass
import time
import numpy as np
import random
from torch.utils.data.dataloader import DataLoader
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import math
import wandb 


rank_1 = 32
rank_2 =  16
alpha_1 = 16
alpha_2 = 8
adapter_name = "default"
dropout = 0.1
target_modules = [
                    'q_proj',
                    'k_proj',
                    'v_proj',
                    'o_proj'
                    #'gate_proj',
                    #'up_proj',
                    #'down_proj'
]
is_second_layar_being_trained = False
is_first_layer_being_trained = True
is_first_layer_being_used_for_inference = True
is_second_layer_being_used_for_inference = False

cache_dir = "/scratch/tathagato"
checkpoint_directory = "/scratch/tathagato/test_cascaded_lora"
base_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
max_sequence_length = 2048
num_proc = 9
#generation config 
max_new_tokens = 400
top_k = 50
top_p = 0.95
temperature = 1
attribute_1 = "length"
attribute_2 = "extractiveness"
test_dataset_path = "../dataset/macdoc/test_dataset.json"
output_directory = "/scratch/tathagato/test_cascaded_lora_outputs"
debug = True
batch_size = 6

#actual dataset config
test_dataset_size_1 = -1 #-1 for all , 1 denotes for first attribute
test_dataset_size_2 = -1

debug = True
if debug:
    test_dataset_size_1 = 120
    test_dataset_size_2 = 120







#model config 
max_seq_length = 2048
max_sequence_length = 2048

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def apply_inference_chat_template(
        example, 
        tokenizer,
    ):
    
    messages = example["messages"]
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system",
                            "content": "You are a friendly chatbot who always help the user"
                                })
    #remove the assistant part for the inference type
    messages = messages[:-1]
    example["messages_for_inference"] = tokenizer.apply_chat_template(messages, add_generation_prompt = True,tokenize=False, truncation = True)
    #tokenized_example = tokenizer.apply_chat_template(messages, add_generation_prompt = True,tokenize=True, return_tensors="pt", truncation = True)
    tokenized_example = tokenizer(example['messages_for_inference'], add_special_tokens = False, return_tensors="pt", truncation = True, max_length = max_sequence_length)
    for key in tokenized_example:
        example[key] = tokenized_example[key]
    return example

# Function to set all seeds for reproducibility
def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)
    # Ensure deterministic behavior by setting environment variables
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CascadedLoRALinear4bit(torch.nn.Module):
    def __init__(self, linear, in_dim, out_dim, rank_1 = rank_1, rank_2 = rank_2, alpha_1 = alpha_1, alpha_2 = alpha_2, adapter_name = "default" , dropout = dropout):
        super().__init__()
        self.base_layer = linear
        std_dev_1 = 1 / torch.sqrt(torch.tensor(rank_1).float())
        std_dev_2 = 1 / torch.sqrt(torch.tensor(rank_2).float())
        if dropout is not None:
            self.lora_dropout = nn.ModuleDict(
                {
                    adapter_name : torch.nn.Dropout(dropout)
                }
            )
        #first dimension
        self.lora_A = nn.ModuleDict(
            {
                adapter_name : torch.nn.Linear(in_dim, rank_1, bias = False)
            }
        )
        self.lora_B = nn.ModuleDict(
            {
                adapter_name : torch.nn.Linear(rank_1, out_dim, bias = False)
            }
        )
        self.lora_A[adapter_name].weight = torch.nn.Parameter(torch.randn(rank_1, in_dim) * std_dev_1)
        self.lora_B[adapter_name].weight = torch.nn.Parameter(torch.zeros(out_dim, rank_1))  
        # B is (rank_1, out_dim)  there B is given B1 and B2 such B1 is (rank_1, rank_2) and B2 is (rank_2, out_dim)
        # A is (in_dim, rank_1) there A is given A1 and A2 such A1 is (in_dim, rank_2) and A2 is (rank_2, rank_1)
        # final output is BA(X) = B2(B1(A2(A1(X))))

        self.lora_A1 = nn.ModuleDict(
            {
                adapter_name : torch.nn.Linear(in_dim, rank_2, bias = False)
            }
        )
        self.lora_A2 = nn.ModuleDict(
            {
                adapter_name : torch.nn.Linear(rank_2, rank_1, bias = False)
            }
        )
        self.lora_B1 = nn.ModuleDict(
            {
                adapter_name : torch.nn.Linear(rank_1, rank_2, bias = False)
            }
        )
        self.lora_B2 = nn.ModuleDict(
            {
                adapter_name : torch.nn.Linear(rank_2, out_dim, bias = False)
            }
        )
        
        self.lora_A1[adapter_name].weight = torch.nn.Parameter(torch.randn(rank_2, in_dim) * std_dev_2)
        self.lora_A2[adapter_name].weight = torch.nn.Parameter(torch.randn(rank_1, rank_2)) 
        self.lora_B1[adapter_name].weight = torch.nn.Parameter(torch.randn(rank_2, rank_1))
        self.lora_B2[adapter_name].weight = torch.nn.Parameter(torch.zeros(out_dim, rank_2) * std_dev_2)  
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.rank_1 = rank_1
        self.rank_2 = rank_2
        #all adapters being used for inference by default and none of them are being trained
        self.is_second_layar_being_trained = is_second_layar_being_trained
        self.is_first_layer_being_trained = is_first_layer_being_trained
        self.is_first_layer_being_used_for_inference = is_first_layer_being_used_for_inference
        self.is_second_layer_being_used_for_inference = is_second_layer_being_used_for_inference
        self.scaling_1 = self.rank_1 / self.alpha_1
        self.scaling_2 = self.rank_2 / self.alpha_1
        self.adapter_name = adapter_name
        

    #https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py
    def forward(self, x):
        

        #self.set_gradients_for_all_layer()
        if x.dtype != torch.float32:
            x = x.float()
        if self.is_first_layer_being_used_for_inference and self.is_second_layer_being_used_for_inference:
            print(x.dtype)
            print(self.base_layer.weight.dtype)
            print(self.lora_A[self.adapter_name].weight.dtype)
            print(self.lora_B[self.adapter_name].weight.dtype)
            print(self.lora_A1[self.adapter_name].weight.dtype)
            #print("first and second both")
            #x = self.scaling_1 * (x @ self.W1_a @ self.W1_b) + self.scaling_2 * (x @ self.W2_a1 @ self.W2_a2 @ self.W2_b1 @ self.W2_b2)
            x  = self.base_layer(x) + self.scaling_1 *  self.lora_B[self.adapter_name](self.lora_A[self.adapter_name](x)) + self.scaling_2 * self.lora_B2[self.adapter_name](self.lora_B1[self.adapter_name](self.lora_A2[self.adapter_name](self.lora_A1[self.adapter_name](x))))
        elif self.is_first_layer_being_used_for_inference and not self.is_second_layer_being_used_for_inference:
            #x = self.scaling_2 * (x @ self.W2_a1 @ self.W2_a2) 
            #print("first only")
            
            # print(self.base_layer.weight.dtype)
            # print(self.lora_A[self.adapter_name].weight.dtype)
            # print(self.lora_B[self.adapter_name].weight.dtype)
            # print(self.lora_A1[self.adapter_name].weight.dtype)
            # print(x.shape)
            # print(self.base_layer(x).shape)
            # print(self.base_layer.weight.shape)
            # print(self.lora_B[self.adapter_name].weight.shape)
            # print(self.lora_A[self.adapter_name].weight.shape)
            # x  =  self.base_layer(x)  
            # y = self.lora_A[self.adapter_name](x)
            # print(y.shape)
            # print("---------------")
            y = self.base_layer(x)
            x  = y + self.scaling_1 * self.lora_B[self.adapter_name](self.lora_A[self.adapter_name](x))

        elif not self.is_first_layer_being_used_for_inference and self.is_second_layer_being_used_for_inference:
            #x = self.scaling_1 * (x @ self.W1_a @ self.W1_b) 
            #print("second only")
            x  = self.base_layer(x) + self.scaling_2 * self.lora_B2[self.adapter_name](self.lora_B1[self.adapter_name](self.lora_A2[self.adapter_name](self.lora_A1[self.adapter_name](x))))
        else:
            #print("none")
            x = self.base_layer(x)
        #print(print_gpu_memory_usage())
        if x.dtype != torch.float16:
            x = x.half()
        return x
def set_gradient_for_all_layers(model, base_layer = False, first_adapter_layer = is_first_layer_being_trained, second_adapter_layer = is_second_layar_being_trained):
    #print(base_layer, first_adapter_layer, second_adapter_layer)
    for name, param in model.named_parameters() :
        
        #what is isinstance of module 
        #print(name)

        if "lora_A1" in name or "lora_A2" in name or "lora_B1" in name or "lora_B2" in name:
            #print("is a second adapter layer")
            if second_adapter_layer:
                param.requires_grad = True
                #print("setting second adapter layer to trainable")
            else:
                param.requires_grad = False
                #print("setting second adapter layer to non trainable")
        elif "lora_A" in name or "lora_B" in name :
            #print("is a first adapter layer")
            if first_adapter_layer:
                param.requires_grad = True
                #print("setting first adapter layer to trainable")
            else:
                param.requires_grad = False
                #print("setting first adapter layer to non trainable")
        else:
            #print("is a base layer")
            if base_layer:
                param.requires_grad = True
                #print("setting base layer to trainable")
            else:
                param.requires_grad = False
                #print("setting base layer to non trainable")
            #print("\n")
    return model

def set_inference_parameters_for_all_layers(model, first_adapter_layer = True, second_adapter_layer = True):
    for name, layer in model.named_modules() :
        #print the class type of the module
        #print(type(module))
        
        # if type is CascadedLoRALinear4bit
        if isinstance(layer, CascadedLoRALinear4bit):
            if first_adapter_layer:
                layer.is_first_layer_being_used_for_inference = True
            else:
                layer.is_first_layer_being_used_for_inference = False
            if second_adapter_layer:
                layer.is_second_layer_being_used_for_inference = True
            else:
                layer.is_second_layer_being_used_for_inference = False
    return model

def print_all_inference_parameters(model):
    for name, layer in model.named_modules() :
        if isinstance(layer, CascadedLoRALinear4bit):
            print(f"Layer: {name}")
            print(f"First Layer being used for inference: {layer.is_first_layer_being_used_for_inference}")
            print(f"Second Layer being used for inference: {layer.is_second_layer_being_used_for_inference}")
            print("-" * 50)
            

def replace_with_cascaded_lora(module, target_modules = target_modules, rank_1 = 64, rank_2 = 32, alpha_1 = 16 , alpha_2 = 16 , adapter_name = "default" , dropout = None):
    for name, child in module.named_children():
        if isinstance(child, bnb.nn.Linear4bit) and name in target_modules:
            setattr(module, name, CascadedLoRALinear4bit(child, child.in_features, child.out_features, rank_1, rank_2, alpha_1, alpha_2, adapter_name , dropout = dropout))
            #put everything in device 
        else:
            replace_with_cascaded_lora(child, target_modules, rank_1, rank_2, alpha_1, alpha_2, adapter_name , dropout = None)

# Function to ensure all submodules are on GPU
def move_to_device(model, device):
    for name, module in model.named_modules():
        try:
            # Check if the module is already on the device
            param = next(module.parameters())
            if param.device != device:
                module.to(device)
        except StopIteration:
            # No parameters in the module
            pass
        

def generate(model, tokenizer, input_text, max_new_tokens = 150, top_k = 50, top_p = 0.95, temperature = 1):
    input_id = tokenizer(input_text, padding = 'max_length' , max_return_tensors="pt")["input_ids"]
def create_cascaded_lora_model_from_quantized_model(quantized_model, target_modules = target_modules, device = None):
    if device is None:
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
#print(quantized_model)

    replace_with_cascaded_lora(quantized_model, target_modules = target_modules, rank_1 = rank_1, rank_2 = rank_2, alpha_1 = alpha_1, alpha_2 = alpha_2, adapter_name = adapter_name , dropout = dropout)
    move_to_device(quantized_model, torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu"))

if __name__ == "__main__":
    #initiate argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="raw_model.pth", help="Path to the model checkpoint")

    args = parser.parse_args()

    model_path = os.path.join(checkpoint_directory,args.model_path)
    print(model_path)
    set_all_seeds(seed = 42)

    os.makedirs(output_directory, exist_ok = True)
    nf4_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_use_double_quant= True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        device_map=f'cuda:0',
        cache_dir = cache_dir,
        torch_dtype = torch.float16,
        attn_implementation = "eager",
        quantization_config = nf4_config, 
    )
    model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path,cache_dir = cache_dir)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'right'

    create_cascaded_lora_model_from_quantized_model(model)
    #load the model
    result = model.load_state_dict(torch.load(model_path))
    # Check for missing or unexpected keys
    if len(result.missing_keys) == 0 and len(result.unexpected_keys) == 0:
        print("State dictionary loaded successfully!")
    else:
        if len(result.missing_keys) > 0:
            print("Missing keys:", result.missing_keys)
        if len(result.unexpected_keys) > 0:
            print("Unexpected keys:", result.unexpected_keys)
    model.eval()

    #setup the datasets
    test_dataset_1 = load_dataset_from_path_phi(test_dataset_path, attribute_1)
    test_dataset_2 = load_dataset_from_path_phi(test_dataset_path, attribute_2)
    if test_dataset_size_1 != -1:
        test_dataset_1 = test_dataset_1.select(range(test_dataset_size_1))
    if test_dataset_size_2 != -1:
        test_dataset_2 = test_dataset_2.select(range(test_dataset_size_2))
    print("test dataset 1 size", len(test_dataset_1))
    print("test dataset 2 size", len(test_dataset_2))
    
    test_column_names_1 = []
    test_column_names_2 = []
    #use for generation
    
    processed_test_dataset_1 = test_dataset_1.map(
        apply_inference_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=test_column_names_1,
        desc="Applying chat template to test_sft",
    )
    processed_test_dataset_2 = test_dataset_2.map(
        apply_inference_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=test_column_names_2,
        desc="Applying chat template to test_sft",
    )


    #filter all examples where input prompt tokenized length is greater than max_sequence_length - max_new_tokens
    processed_test_dataset_1 = processed_test_dataset_1.filter(lambda x: len(x["input_ids"][0]) < max_sequence_length - (max_new_tokens + 5))
    processed_test_dataset_2 = processed_test_dataset_2.filter(lambda x: len(x["input_ids"][0]) < max_sequence_length - (max_new_tokens + 5))

    #do generate on the first attribute
    print("after filtering")
    print("processed_test_dataset_1 size", len(processed_test_dataset_1))
    print("processed_test_dataset_2 size", len(processed_test_dataset_2))

    #do batching over the dataset 
    # processed_test_dataset_1.set_format(type="torch", columns=["input_ids", "attention_mask"])
    # processed_test_dataset_2.set_format(type="torch", columns=["input_ids", "attention_mask"])
    # processed_test_dataset_1 = DataLoader(processed_test_dataset_1, batch_size = 4, shuffle = False)
    # processed_test_dataset_2 = DataLoader(processed_test_dataset_2, batch_size = 4, shuffle = False)
    set_inference_parameters_for_all_layers(model, first_adapter_layer = True, second_adapter_layer = False)


    start_time = time.time()
    first_output_dict = {}
    second_output_dict = {}
    model.eval()
    tokenizer.pad_token = tokenizer.unk_token
    #https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
    tokenizer.padding_side = 'left'
    batch_size = 6
    with torch.no_grad():
        texts = []
        examples = []
        for index, example in tqdm.tqdm(enumerate(processed_test_dataset_1), desc = 'first attribute inference'):
            examples.append(example)
            texts.append(example["messages_for_inference"])
            if len(texts) == batch_size or index == len(processed_test_dataset_1) - 1:
                input_ids = tokenizer(texts, padding = 'max_length', max_length = max_sequence_length, return_tensors = "pt").to("cuda")
                #print(input_ids['input_ids'].shape)

                generation_output = model.generate(input_ids["input_ids"], max_new_tokens = max_new_tokens, num_return_sequences = 1, top_k = top_k, top_p = top_p, temperature = temperature, do_sample = True, return_dict_in_generate = True)
                decode_output = tokenizer.batch_decode(generation_output["sequences"], skip_special_tokens = True)
                for i, text in enumerate(texts):
                    summary = decode_output[i].split("\n")[-1]
                    cur_index = len(first_output_dict)
                    first_output_dict[cur_index] = {}
                    for key in example:
                        first_output_dict[cur_index][key] = example[key]
                    first_output_dict[cur_index]["generation_output"] = decode_output[i]
                    first_output_dict[cur_index]["summary"] = summary
                texts = []
                examples = []
    print("Time taken for inference", time.time() - start_time)
    with torch.no_grad():
        texts = []
        examples = []
        for index, example in tqdm.tqdm(enumerate(processed_test_dataset_2), desc = 'second attribute inference'):
            examples.append(example)
            texts.append(example["messages_for_inference"])
            if len(texts) == batch_size or index == len(processed_test_dataset_2) - 1:
                input_ids = tokenizer(texts, padding = 'max_length', max_length = max_sequence_length, return_tensors = "pt").to("cuda")
                #print(input_ids['input_ids'].shape)

                generation_output = model.generate(input_ids["input_ids"], max_new_tokens = max_new_tokens, num_return_sequences = 1, top_k = top_k, top_p = top_p, temperature = temperature, do_sample = True, return_dict_in_generate = True)
                decode_output = tokenizer.batch_decode(generation_output["sequences"], skip_special_tokens = True)
                for i, text in enumerate(texts):
                    summary = decode_output[i].split("\n")[-1]
                    cur_index = len(second_output_dict)
                    second_output_dict[cur_index] = {}
                    for key in example:
                        second_output_dict[cur_index][key] = example[key]
                    second_output_dict[cur_index]["generation_output"] = decode_output[i]
                    second_output_dict[cur_index]["summary"] = summary
                texts = []
                examples = []
    

    # with torch.no_grad():
    #     for index, example in tqdm.tqdm(enumerate(processed_test_dataset_1), desc = 'first attribute inference'):
    #         input_ids = torch.tensor(example["input_ids"]).to(device)
    #         generation_output = model.generate(input_ids, max_new_tokens = max_new_tokens, num_return_sequences = 1, top_k = top_k, top_p = top_p, temperature = temperature, do_sample = True, return_dict_in_generate = True)
    #         decode_output = tokenizer.batch_decode(generation_output["sequences"], skip_special_tokens = True)
    #         summary = decode_output[0].split("\n")[-1]
    #         print(decode_output)
    #         print(summary)
    #         first_output_dict[index] = {}
    #         for key in example:
    #             first_output_dict[index][key] = example[key]
    #         first_output_dict[index]["generation_output"] = decode_output
    #         first_output_dict[index]["summary"] = summary
    # print("Time taken for inference", time.time() - start_time)

    # #second attribute inference
    # set_inference_parameters_for_all_layers(model, first_adapter_layer = True, second_adapter_layer = True)
    # start_time = time.time()
    # second_output_dict = {}
    # model.eval()
    # with torch.no_grad():
    #     for index, example in tqdm.tqdm(enumerate(processed_test_dataset_2), desc = 'second attribute inference'):
    #         input_ids = torch.tensor(example["input_ids"]).to(device)
    #         generation_output = model.generate(input_ids, max_new_tokens = max_new_tokens, num_return_sequences = 1, top_k = top_k, top_p = top_p, temperature = temperature, do_sample = True, return_dict_in_generate = True)
    #         decode_output = tokenizer.batch_decode(generation_output["sequences"], skip_special_tokens = True)
    #         summary = decode_output[0].split("\n")[-1]
    #         print(decode_output)
    #         print(summary)
    #         second_output_dict[index] = {}
    #         for key in example:
    #             second_output_dict[index][key] = example[key]
    #         second_output_dict[index]["generation_output"] = decode_output
    #         second_output_dict[index]["summary"] = summary
    # print("Time taken for inference", time.time() - start_time)
    final_output = {"first attribute": first_output_dict, "second attribute": second_output_dict}
    print(final_output)
    with open(os.path.join(output_directory, os.path.basename(args.model_path) + ".json"), "w") as f:
        json.dump(final_output, f)






    #filter all examples where 


    #do eval on the first attribute
    
    #print_all_inference_parameters(model)
    #print_all_inference_parameters(model)

    


