import sys
import logging
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit
import datasets
from trl import SFTTrainer
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
import safetensors
import torch.nn as nn
from functools import partial
import tqdm
from datasets import load_dataset
from peft import LoraConfig,PeftConfig, PeftModel, PeftModelForCausalLM
import torch
import transformers
from trl import SFTTrainer
import argparse
from datetime import datetime
import os
from utils import load_dataset_from_path_phi, load_multi_attribute_dataset_from_path,print_trainable_parameters, count_model_parameters
from transformers import pipeline
import json

#config 
rank_1 = 64
rank_2 = 32
alpha_1 = 16
alpha_2 = 16
adapter_name = "default"
dropout = 0.1

target_modules = [
                    'q_proj',
                    'k_proj',
                    'v_proj',
                    'o_proj',
                    'gate_proj',
                    'up_proj',
                    'down_proj'
]

#run command accelerate launch cascaded_lora.py
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
        self.lora_A2[adapter_name].weight = torch.nn.Parameter(torch.zeros(rank_1, rank_2))
        self.lora_B1[adapter_name].weight = torch.nn.Parameter(torch.zeros(rank_2, rank_1))
        self.lora_B2[adapter_name].weight = torch.nn.Parameter(torch.zeros(out_dim, rank_2) * std_dev_2)  
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.rank_1 = rank_1
        self.rank_2 = rank_2
        self.is_second_layar_being_trained = False
        self.is_first_layer_being_trained = False
        self.is_first_layer_being_used_for_inference = True
        self.is_first_layer_being_used_for_inference = True
        self.scaling_1 = self.rank_1 / self.alpha_1
        self.scaling_2 = self.rank_2 / self.alpha_1
        self.adapter_name = adapter_name
        self.freeze_base_layer()
        self.freeze_the_second_adapter()
        self.tune_the_first_adapter()


    def freeze_base_layer(self):
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def set_gradients_for_all_layer(self):
        if self.is_second_layar_being_trained:
            self.lora_A1[self.adapter_name].requires_grad = True
            self.lora_A2[self.adapter_name].requires_grad = True
            self.lora_B1[self.adapter_name].requires_grad = True
            self.lora_B2[self.adapter_name].requires_grad = True


        else:
            self.lora_A1[self.adapter_name].requires_grad = False
            self.lora_A2[self.adapter_name].requires_grad = False
            self.lora_B1[self.adapter_name].requires_grad = False
            self.lora_B2[self.adapter_name].requires_grad = False
            
        if self.is_first_layer_being_trained:
            self.lora_A[self.adapter_name].requires_grad = True
            self.lora_B[self.adapter_name].requires_grad = True
        else:
            self.lora_A[self.adapter_name].requires_grad = False
            self.lora_B[self.adapter_name].requires_grad = False  
    
    def tune_the_first_adapter(self):
        self.is_first_layer_being_trained = True
    
    def freeze_the_first_adapter(self):
        self.is_first_layer_being_trained = False
    
    def tune_the_second_adapter(self):
        self.is_second_layar_being_trained = True
    
    def freeze_the_second_adapter(self):
        self.is_second_layar_being_trained = False

    def forward(self, x):

        self.set_gradients_for_all_layer()
        if self.is_first_layer_being_used_for_inference and self.is_second_layer_being_used_for_inference:
            #x = self.scaling_1 * (x @ self.W1_a @ self.W1_b) + self.scaling_2 * (x @ self.W2_a1 @ self.W2_a2 @ self.W2_b1 @ self.W2_b2)
            output  = self.linear(x) + self.scaling_1 * (self.W1['A'](self.W1['B'](x))) + self.scaling_2 * (self.W2['B2'](self.W2['A2'](self.W2['B1'](self.W2['A1'](x)))))
        if self.is_first_layer_being_used_for_inference and not self.is_second_layer_being_used_for_inference:
            #x = self.scaling_2 * (x @ self.W2_a1 @ self.W2_a2) 
            output  =  self.linear(x)  + self.scaling_1 * (self.W1['A'](self.W1['B'](x))) 
        return output
    
def replace_with_cascaded_lora(module, target_modules = target_modules, rank_1 = 64, rank_2 = 32, alpha_1 = 16 , alpha_2 = 16 , adapter_name = "default" , dropout = None):
    for name, child in module.named_children():
        if isinstance(child, bnb.nn.Linear4bit) and name in target_modules:
            #setattr(module, name, CascadedLoRALinear4bit(child, in_dim, out_dim, **kwargs))
            #print(name)
            #print(child.in_features, child.out_features)
            #get the device of the child
            setattr(module, name, CascadedLoRALinear4bit(child, child.in_features, child.out_features, rank_1, rank_2, alpha_1, alpha_2, adapter_name , dropout = dropout))
            #put everything in device 
        else:
            replace_with_cascaded_lora(child, target_modules, rank_1, rank_2, alpha_1, alpha_2, adapter_name , dropout = None)
def print_device_and_dtype(model, file = sys.stdout):
    if file == sys.stdout:
            for name, module in model.named_modules():
            # Get the device and dtype of the module's parameters
            #file = open(file, "a")
                try:
                    param = next(module.parameters())
                    device = param.device
                    dtype = param.dtype
                    type = param.type()
                except StopIteration:
                    device = 'No parameters'
                    dtype = 'No parameters'
                    type = 'No parameters'

                
                # Print the name, device, and dtype of the module
                print(f"Module: {name}", file = file)
                print(f"  Device: {device}", file = file)
                print(f"  Dtype: {dtype}", file = file)
                print(f"  Type: {type}", file = file)
                print(" ",file = file )
            return 

    with open(file, "w") as file:

        for name, module in model.named_modules():
            # Get the device and dtype of the module's parameters
            #file = open(file, "a")
            try:
                param = next(module.parameters())
                device = param.device
                dtype = param.dtype
                type = param.type()
            except StopIteration:
                device = 'No parameters'
                dtype = 'No parameters'
                type = 'No parameters'

            
            # Print the name, device, and dtype of the module
            print(f"Module: {name}", file = file)
            print(f"  Device: {device}", file = file)
            print(f"  Dtype: {dtype}", file = file)
            print(f"  Type: {type}", file = file)
            print(" ",file = file )
# Function to ensure all submodules are on GPU
def move_to_device(model, device):
    for name, module in model.named_modules():
        try:
            # Check if the module is already on the device
            param = next(module.parameters())
            if param.device != device:
                # Move the module to the specified device
                module.to(device)
                #print(f"Moved module: {name} to {device}")
        except StopIteration:
            # No parameters in the module
            pass


def create_cascaded_lora_model_from_quantized_model(quantized_model, target_modules = target_modules, device = None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(quantized_model)

    replace_with_cascaded_lora(quantized_model, target_modules = target_modules, rank_1 = rank_1, rank_2 = rank_2, alpha_1 = alpha_1, alpha_2 = alpha_2, adapter_name = adapter_name , dropout = dropout)
    move_to_device(quantized_model, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    #print_device_and_dtype(quantized_model, file = "cascaded_lora_structure.txt")
def apply_inference_chat_template(
        example, 
        tokenizer,
    ):
    
    messages = example["messages"]
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system",
                            "content": "You are a friendly chatbot who always responds in the style of a pirate"
                                })
    #make assistant part empty
    messages[-1]["content"] = ""
    example["messages_for_inference"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return example
def apply_chat_template(
    example,
    tokenizer,
):
    messages = example["messages"]
    # Add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system",
                            "content": "You are a friendly chatbot who always responds in the style of a pirate"
                                })
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False)
    return example
def compare_state_dicts(initial_dict, final_dict):
    modified_layers = []
    for layer_name in initial_dict:
        if not torch.equal(initial_dict[layer_name], final_dict[layer_name]):
            modified_layers.append(layer_name)
    return modified_layers

if __name__ == "__main__":
    output_directory = "test_cascaded_lora"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    training_config = {
        "bf16": False,
        "fp16" : True,
        "do_eval": False,
        "learning_rate": 5e-4,
        "log_level": "info",
        "logging_steps": 20,
        "logging_strategy": "steps",
        "lr_scheduler_type": "cosine",
        "num_train_epochs": 1,
        "max_steps": -1,
        "output_dir": output_directory,
        "overwrite_output_dir": True,
        "per_device_eval_batch_size": 1,
        "per_device_train_batch_size": 1,
        "remove_unused_columns": False,
        "save_steps": 200,
        "save_total_limit": 400,
        "seed": 0,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs":{"use_reentrant": False},
        "gradient_accumulation_steps": 2,
        "warmup_ratio": 0.2,
        }
    train_conf = TrainingArguments(**training_config)

    
    nf4_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_use_double_quant= True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='cuda:0',
        cache_dir = "/scratch/tathagato",
        attn_implementation = "eager",
        quantization_config = nf4_config, 
    )
    print("loading quantized model")
    base_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path,cache_dir = "/scratch/tathagato")

    #cascaded lora model
    print("creating cascaded lora model")
    create_cascaded_lora_model_from_quantized_model(model)
    print("model created")
    tokenizer.padding_side = 'right'
    attribute = "length"
    train_dataset_path = "/home2/tathagato/summarization/MACSum/dataset/macdoc/train_dataset.json"
    train_dataset = load_dataset_from_path_phi(train_dataset_path, attribute)
    train_dataset = train_dataset.select(range(100))
    column_names = []

    processed_train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to train_sft",
    )
    
    #initial_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    trainer = SFTTrainer(
        model=model,
        args=train_conf,
        train_dataset=processed_train_dataset,
        max_seq_length=2048,
        dataset_text_field="text",
        tokenizer=tokenizer,
        #callbacks=[InferenceCallback(model, tokenizer, processed_test_dataset, inference_directory=args.inference_directory, args=args)],
        packing=True

    )
    trainer.train()
    #final_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    #modified_layers = compare_state_dicts(initial_state_dict, final_state_dict)
    with open("modified_layers.txt", "w") as file:
        for layer in modified_layers:
            file.write(layer + "\n")
    








    





