#some code borrowed from https://github.com/jzhang38/TinyLlama/blob/main/sft/finetune.py
#some code inspired/borrowed/modified from https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
#some code borrowed from https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ch07.ipynb
#https://discuss.huggingface.co/t/instruction-tuning-llm/67597/8
#according to the link above labels and input_ids are same and the shift right part is part of 
#of the model itself. So labels and input ids should be same
#pad ids should be -100 to be ignored in the loss computation
#padded dataset is not a great idea for sft as examples are broken across batches and instances
#hence the instruction can be one one data point and the article can be in another data point
#so we shouldn't do padded dataset. Instead we should do non packed dataset with one example 
#per data point

import sys
import logging
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit
import datasets
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
import argparse
from datetime import datetime
import os
from utils import load_dataset_from_path_phi, load_multi_attribute_dataset_from_path,print_trainable_parameters, count_model_parameters
from transformers import pipeline
import json
from dataclasses import dataclass

# QUESTIONS :
# when doing instruction finetuning for summarization 
# lets say we have a instruction like "summarize the text" 
# input : article
# output : summary
# is the loss unsupervised loss on (instruction + article + summary)
# is the loss supervised loss on loss = (summary , model(instruction + article))
# because sft trainer asks for a string which is (instruction + article + summary)


#TODO
#all these config should be overwritable from the command line
rank_1 = 64
rank_2 =  32
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

cache_dir = "/scratch/tathagato"
output_directory = "test_cascaded_lora"
base_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
train_dataset_path = "/home2/tathagato/summarization/MACSum/dataset/macdoc/train_dataset.json"
batch_size = 2
max_sequence_length = 2048
num_proc = 9


#con


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"percentage: {trainable_params/all_param*100:.2f}%"
    )

#transform a linear layer to a cascaded lora layer
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
        #all adapters being used for inference by default and none of them are being trained
        self.is_second_layar_being_trained = False
        self.is_first_layer_being_trained = False
        self.is_first_layer_being_used_for_inference = True
        self.is_first_layer_being_used_for_inference = True
        self.scaling_1 = self.rank_1 / self.alpha_1
        self.scaling_2 = self.rank_2 / self.alpha_1
        self.adapter_name = adapter_name
        
        #initialize with only the  adapter being trained
        # 1. freeze base model
        # 2. freeze the second adapter
        # 3. tune the first adapter
    #     self.freeze_base_layer()
    #     self.freeze_the_second_adapter()
    #     self.tune_the_first_adapter()
    #     self.set_gradients_for_all_layer()



    # def freeze_base_layer(self):
    #     for param in self.base_layer.parameters():
    #         param.requires_grad = False

    # def set_gradients_for_all_layer(self):
    #     #print("setting gradients for all layers")
    #     #print(self.is_second_layar_being_trained)
    #     if self.is_second_layar_being_trained:
    #         self.lora_A1[self.adapter_name].requires_grad = True
    #         self.lora_A2[self.adapter_name].requires_grad = True
    #         self.lora_B1[self.adapter_name].requires_grad = True
    #         self.lora_B2[self.adapter_name].requires_grad = True


    #     else:
    #         self.lora_A1[self.adapter_name].requires_grad = False
    #         self.lora_A2[self.adapter_name].requires_grad = False
    #         self.lora_B1[self.adapter_name].requires_grad = False
    #         self.lora_B2[self.adapter_name].requires_grad = False
            
    #     if self.is_first_layer_being_trained:
    #         self.lora_A[self.adapter_name].requires_grad = True
    #         self.lora_B[self.adapter_name].requires_grad = True
    #     else:
    #         self.lora_A[self.adapter_name].requires_grad = False
    #         self.lora_B[self.adapter_name].requires_grad = False  
    #     #print("setting gradients for all layers done")
    #     #print(self.lora_A1[self.adapter_name].requires_grad, self.lora_A2[self.adapter_name].requires_grad, self.lora_B1[self.adapter_name].requires_grad, self.lora_B2[self.adapter_name].requires_grad)
    
    # def tune_the_first_adapter(self):
    #     self.is_first_layer_being_trained = True
    
    # def freeze_the_first_adapter(self):
    #     self.is_first_layer_being_trained = False
    
    # def tune_the_second_adapter(self):
    #     self.is_second_layar_being_trained = True
    
    # def freeze_the_second_adapter(self):
    #     self.is_second_layar_being_trained = False

    def forward(self, x):

        #self.set_gradients_for_all_layer()
        if self.is_first_layer_being_used_for_inference and self.is_second_layer_being_used_for_inference:
            #x = self.scaling_1 * (x @ self.W1_a @ self.W1_b) + self.scaling_2 * (x @ self.W2_a1 @ self.W2_a2 @ self.W2_b1 @ self.W2_b2)
            output  = self.linear(x) + self.scaling_1 * (self.W1['A'](self.W1['B'](x))) + self.scaling_2 * (self.W2['B2'](self.W2['A2'](self.W2['B1'](self.W2['A1'](x)))))
        elif self.is_first_layer_being_used_for_inference and not self.is_second_layer_being_used_for_inference:
            #x = self.scaling_2 * (x @ self.W2_a1 @ self.W2_a2) 
            output  =  self.linear(x)  + self.scaling_1 * (self.W1['A'](self.W1['B'](x))) 
        elif not self.is_first_layer_being_used_for_inference and not self.is_second_layer_being_used_for_inference:
            #x = self.linear(x)
            output  = self.linear(x)
        
        return output

def set_gradient_for_all_layers(model, base_layer = False, first_adapter_layer = True, second_adapter_layer = False):
    for name, param in model.named_parameters() :
            #what is isinstance of module 
            #print(name)

            if "lora_A1" in name or "lora_A2" in name or "lora_B1" in name or "lora_B2" in name:
                #print("is a second adapter layer")
                if second_adapter_layer:
                    # for param in module.parameters():
                    #     param.requires_grad = True
                    param.requires_grad = True
                    #print("setting second adapter layer to trainable")
                else:
                    # for param in module.parameters():
                    #     param.requires_grad = False
                    param.requires_grad = False
                    #print("setting second adapter layer to non trainable")
            elif "lora_A" in name or "lora_B" in name :
                #print("is a first adapter layer")

                if first_adapter_layer:
                    # for param in module.parameters():
                    #     param.requires_grad = True
                    param.requires_grad = True
                    #print("setting first adapter layer to trainable")
                else:
                    # for param in module.parameters():
                    #     param.requires_grad = False
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
            for name, param in model.named_parameters():
            # Get the device and dtype of the module's parameters
            #file = open(file, "a")
                #if not isinstance(module, (nn.ModuleList, nn.ModuleDict)) and hasattr(module, 'weight') and module.weight is not None:
                if True:

                    try:
                        #param = next(module.parameters())
                        device = param.device
                        dtype = param.dtype
                        type = param.type()
                        req_grad = param.requires_grad
                        #check if module dict 
                        if isinstance(param, nn.ModuleDict):
                            is_module_dict = True
                        else:
                            is_module_dict = False
                    except StopIteration:
                        device = 'No parameters'
                        dtype = 'No parameters'
                        type = 'No parameters'

                    
                    # Print the name, device, and dtype of the module
                    print(f"Module: {name}", file = file)
                    print(f"  Device: {device}", file = file)
                    print(f"  Dtype: {dtype}", file = file)
                    print(f"  Type: {type}", file = file)
                    print(f"  Requires Grad: {req_grad}", file = file)
                    print(f"  Is Module Dict: {is_module_dict}", file = file)
                    print(" ",file = file )
            return 

    with open(file, "w") as file:

        for name, module in model.named_modules():
            if not isinstance(module, (nn.ModuleList, nn.ModuleDict)) and hasattr(module, 'weight') and module.weight is not None: 

                # Get the device and dtype of the module's parameters
                #file = open(file, "a")
                try:
                    param = next(module.parameters())
                    device = param.device
                    dtype = param.dtype
                    type = param.type()
                    req_grad = param.requires_grad
                    #check if module dict 
                    if isinstance(module, nn.ModuleDict):
                        is_module_dict = True
                    else:
                        is_module_dict = False
                except StopIteration:
                    device = 'No parameters'
                    dtype = 'No parameters'
                    type = 'No parameters'

                
                print(f"Module: {name}", file = file)
                print(f"  Device: {device}", file = file)
                print(f"  Dtype: {dtype}", file = file)
                print(f"  Type: {type}", file = file)
                print(f"  Requires Grad: {req_grad}", file = file)
                print(f"  Is Module Dict: {is_module_dict}", file = file)
                print(" ",file = file )

#line 379 https://github.com/huggingface/trl/blob/18a33ffcd3a576f809b6543a710e989333428bd3/trl/trainer/sft_trainer.py#L379
# this does not return batched dataset and each example is tokenized but do not have the same length
def prepare_non_packed_dataloader(tokenizer, dataset, dataset_text_field = "text", max_seq_length = max_sequence_length, add_special_tokens=True):
    # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
    def tokenize(element):
        outputs = tokenizer(
            element[dataset_text_field],
            add_special_tokens=add_special_tokens,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}
    tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            num_proc=num_proc,
            batch_size=batch_size,
        )

    return tokenized_dataset

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

#TODO 
#should create a new class like PEFTModel does and make it part of the model class
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
                            "content": "Write a response that appropriately completes the request"
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
                            "content": "Write a response that appropriately completes the request"
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
        torch_dtype=torch.bfloat16,
        device_map='cuda:0',
        cache_dir = cache_dir,
        attn_implementation = "eager",
        quantization_config = nf4_config, 
    )

    print("loading quantized model")
    model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path,cache_dir = cache_dir)

    #cascaded lora model
    print("creating cascaded lora model")
    create_cascaded_lora_model_from_quantized_model(model)
    print("model created")
    tokenizer.padding_side = 'right'
    attribute = "length"
    train_dataset = load_dataset_from_path_phi(train_dataset_path, attribute)
    train_dataset = train_dataset.select(range(50))
    column_names = []

    processed_train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to train_sft",
    )
    #TODO
    #1 set gradient for all layer should be made as method for the model class itself

    model = set_gradient_for_all_layers(model, base_layer = False, first_adapter_layer = True, second_adapter_layer = False)
    #print the model structure, individual layer device and dtype and the trainable parameters
    #print_device_and_dtype(model)
    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")
    #     if "base_layer" in name:
    #         print(param.shape)
    #         print(param.dtype)
    #print(model)

    #now the dataset 
    # for example in processed_train_dataset:
    #     print(example["text"])
    #     print("\n")
    #     break
    #dataset is fine in textual form. 
    # I need to tokenize it and then pass it to the model
    # ideally should keep tokenized dataset in bin format
    #get non-packed tokenized dataset
    tokenized_dataset = prepare_non_packed_dataloader(tokenizer, processed_train_dataset, dataset_text_field = "text", max_seq_length = max_sequence_length, add_special_tokens=True)
    for example in tokenized_dataset:
        print(len(example["input_ids"]))
    
        
    
    
