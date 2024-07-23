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

import inspect
import sys
import logging
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit
import datasets
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, set_seed
import safetensors
import torch.nn as nn
from functools import partial
import tqdm
from datasets import load_dataset
from peft import LoraConfig,PeftConfig, PeftModel, PeftModelForCausalLM
import torch
import torch.nn.functional as F
import transformers
from trl import SFTTrainer
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
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
###################################################################################################################################################################
#----------------------------------------------------------------TODOS----------------------------------------------------------------------------------------------#
''' 
1.check there still exist some discrepancy between train and test prompt. That is there should </s> after each role prompt but <s> only in the beginning of the prompt 
that is currently not the case ---> this looks fine I think. FIXED
2. check in generate if model is giving the same output without using the 2 cascading layers. It should be same as base model. ---> IT APPEARS TO BE SAME.
3. check which layers weight change if called loss.backward 
4. when doing instruction finetuning for summarization 
   lets say we have a instruction like "summarize the text" ,input : article
   output : summary
   is the loss unsupervised loss on (instruction + article + summary)
   is the loss supervised loss on loss = (summary , model(instruction + article))
   because sft trainer asks for a string which is (instruction + article + summary)

'''
#some math
''' 
if we have to lora a layer of say in_dim = 5 and out_dim = 3
let rank be 2
then A = (5,2) and B = (2,3)
then the output of the lora layer will be
output = x @ A @ B which has shape (batch_size, 3)
Now if want to represent them as linear layers 
then we will have 
L_A = nn.Linear(2,5)
L_B = nn.Linear(2,3)
x is the input to the layer which has shape (batch_size, 5)
then the output will be 
output = L_B(L_A(x))
L_B(L_A(x))
L
'''


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------#
###################################################################################################################################################################
#config 

accelerator = Accelerator(mixed_precision = 'fp16')
#TODO
#all these config should be overwritable from the command line
rank_1 = 16
alpha_1 = 16
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
is_first_layer_being_trained = True
is_first_layer_being_used_for_inference = True

cache_dir = "/scratch/tathagato"
output_directory = "test_cascaded_lora"
base_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
train_dataset_path = "/home2/tathagato/summarization/MACSum/dataset/macdoc/train_dataset.json"
max_sequence_length = 2048
num_proc = 9
#generation config 
max_new_tokens = 150
top_k = 50
top_p = 0.95
temperature = 1
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = accelerator.device
#dataset config 
attribute = "length"
train_dataset_path = "../dataset/macdoc/train_dataset.json"
test_dataset_path = "../dataset/macdoc/test_dataset.json"
test_dataset_size = 4 #-1 for all
train_dataset_size = 4 #-1 means for all

#model config 
max_seq_length = 2048
max_sequence_length = 2048

#training config
learning_rate = 3e-4
num_train_epochs = 1
batch_size_per_device = 1
num_device = 4
gradient_accumulation_steps = 2
do_gradient_checkpointing = True #wont fit in memory without gradient checkpointing
fp_16 = True
warmup_ratio = 0.2
lr_scheduler_type = "cosine"


# copied from transformers.models.bart.modeling_bart
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`): input ids
        pad_token_id (`int`): The id of the `padding` token.
        decoder_start_token_id (`int`): The id of the `start` token.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

#directly copied from https://github.com/huggingface/peft/blob/v0.11.0/src/peft/utils/other.py#L89
def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None):
    r"""
    Note this method only works for `transformers` models.

    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
        use_gradient_checkpointing (`bool`, *optional*, defaults to `True`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        gradient_checkpointing_kwargs (`dict`, *optional*, defaults to `None`):
            Keyword arguments to pass to the gradient checkpointing function, please refer to the documentation of
            `torch.utils.checkpoint.checkpoint` for more details about the arguments that you can pass to that method.
            Note this is only available in the latest transformers versions (> 4.34.1).
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    is_gptq_quantized = getattr(model, "quantization_method", None) == "gptq"
    is_aqlm_quantized = getattr(model, "quantization_method", None) == "aqlm"
    is_eetq_quantized = getattr(model, "quantization_method", None) == "eetq"
    is_hqq_quantized = getattr(model, "quantization_method", None) == "hqq" or getattr(model, "hqq_quantized", False)

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    if not is_gptq_quantized and not is_aqlm_quantized and not is_eetq_quantized and not is_hqq_quantized:
        # cast all non INT8 parameters to fp32
        for param in model.parameters():
            if (
                (param.dtype == torch.float16) or (param.dtype == torch.bfloat16)
            ) and param.__class__.__name__ != "Params4bit":
                param.data = param.data.to(torch.float32)

    if (
        loaded_in_kbit or is_gptq_quantized or is_aqlm_quantized or is_eetq_quantized or is_hqq_quantized
    ) and use_gradient_checkpointing:
        # When having `use_reentrant=False` + gradient_checkpointing, there is no need for this hack
        if "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]:
            # For backward compatibility
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # To support older transformers versions, check if the model supports gradient_checkpointing_kwargs
        _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
            inspect.signature(model.gradient_checkpointing_enable).parameters
        )

        if not _supports_gc_kwargs and len(gradient_checkpointing_kwargs) > 0:
            warnings.warn(
                "gradient_checkpointing_kwargs is not supported in this version of transformers. The passed kwargs will be ignored."
                " if you want to use that feature, please upgrade to the latest version of transformers.",
                FutureWarning,
            )

        gc_enable_kwargs = (
            {} if not _supports_gc_kwargs else {"gradient_checkpointing_kwargs": gradient_checkpointing_kwargs}
        )

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable(**gc_enable_kwargs)
    return model

def debug_trainable_parameters(model):
    total_params = 0

    #    Print details of layers with requires_grad=True
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()  # Number of elements in the parameter
            total_params += num_params
            print(f"Layer: {name}")
            print(f"Weight Shape: {param.shape}")
            print(f"Device: {param.device}")
            print(f"Number of Parameters in this layer: {num_params}")
            print(f"Running Total Number of Parameters: {total_params}")
            print("-" * 50)

    # Print total number of parameters
    print(f"Total number of trainable parameters: {total_params}")

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
    def __init__(self, linear, in_dim, out_dim, rank_1 = rank_1, alpha_1 = alpha_1, adapter_name = "default" , dropout = dropout):
        super().__init__()
        self.base_layer = linear
        std_dev_1 = 1 / torch.sqrt(torch.tensor(rank_1).float())
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

        self.alpha_1 = alpha_1
        self.rank_1 = rank_1
        #all adapters being used for inference by default and none of them are being trained
        self.is_first_layer_being_trained = is_first_layer_being_trained
        self.is_first_layer_being_used_for_inference = is_first_layer_being_used_for_inference
        self.scaling_1 = self.rank_1 / self.alpha_1
        self.adapter_name = adapter_name


    #https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py
    def forward(self, x):
        if self.is_first_layer_being_used_for_inference:
            #print("first only")
            x  =  self.base_layer(x)  + self.scaling_1 * self.lora_B[self.adapter_name](self.lora_A[self.adapter_name](x))
        else:
            #print("none")
            x = self.base_layer(x)
        #print(print_gpu_memory_usage())
        return x
#https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
def configure_optimizers(model, weight_decay, learning_rate, device_type = "cuda"):
    # start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    #fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    # use_fused = fused_available and device_type == "cuda"
    # if master_process:
    #     print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
    return optimizer

def set_gradient_for_all_layers(model, base_layer = False, first_adapter_layer = is_first_layer_being_trained):
    #print(base_layer, first_adapter_layer, second_adapter_layer)
    for name, param in model.named_parameters() :

        if "lora_A" in name or "lora_B" in name :
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





def replace_with_cascaded_lora(module, target_modules = target_modules, rank_1 = 64, alpha_1 = 16, adapter_name = "default" , dropout = None):
    for name, child in module.named_children():
        if isinstance(child, bnb.nn.Linear4bit) and name in target_modules:
            #setattr(module, name, CascadedLoRALinear4bit(child, in_dim, out_dim, **kwargs))
            #print(name)
            #print(child.in_features, child.out_features)
            #get the device of the child
            setattr(module, name, CascadedLoRALinear4bit(child, child.in_features, child.out_features, rank_1, alpha_1, adapter_name , dropout = dropout))
            #put everything in device 
        else:
            replace_with_cascaded_lora(child, target_modules, rank_1, alpha_1, adapter_name , dropout = None)
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
def count_parameter_bytes(model):
    total_bytes_requires_grad = 0
    total_bytes_no_grad = 0
    
    for param in model.parameters():
        param_bytes = param.numel() * param.element_size()
        if param.requires_grad:
            total_bytes_requires_grad += param_bytes
        else:
            total_bytes_no_grad += param_bytes
    print(f"Total bytes for parameters that require grad in GB: {total_bytes_requires_grad / (1024**3)}")
    print(f"Total bytes for parameters that do not require grad in GB: {total_bytes_no_grad / (1024**3)}")
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

    replace_with_cascaded_lora(quantized_model, target_modules = target_modules, rank_1 = rank_1, alpha_1 = alpha_1, adapter_name = adapter_name , dropout = dropout)
    move_to_device(quantized_model, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    #print_device_and_dtype(quantized_model, file = "cascaded_lora_structure.txt")
    
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
    example["messages_for_inference"] = tokenizer.apply_chat_template(messages, add_generation_prompt = True,tokenize=False)
    return example
def apply_chat_template(
    example,
    tokenizer,
):
    messages = example["messages"]
    # Add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system",
                            "content": "You are a friendly chatbot who always help the user"
                                })
    text = tokenizer.apply_chat_template(
        messages, tokenize=False)
    #you can give pretokenized dataset to sfft also
    tokenized_example = tokenizer(text, padding="max_length", truncation=True, max_length=2048)
    # for key in tokenized_example.keys():
    #     example[key] = tokenized_example[key]
    example = {}
    example['text'] = text
    example["input_ids"] = tokenized_example["input_ids"]
    example["attention_mask"] = tokenized_example["attention_mask"]
    example['labels'] = tokenized_example["input_ids"]
    return example


def estimate_vram_in_gb(model, sample_input, optimizer_cls=torch.optim.AdamW):
    total_memory = 0
    param_memory = 0
    grad_memory = 0
    activation_memory = 0

    # Function to calculate memory for a tensor
    def tensor_memory(tensor):
        return tensor.numel() * tensor.element_size()

    # Calculate memory for parameters and gradients
    for name, param in model.named_parameters():
        param_memory += tensor_memory(param) / (1024**3)  # Convert to GB
        if param.requires_grad:
            grad_memory += tensor_memory(param) / (1024**3)  # Convert to GB
    print(f"Parameter memory: {param_memory:.2f} GB")

    # Calculate memory for activations
    def hook_fn(module, input, output):
        nonlocal activation_memory
        if isinstance(output, torch.Tensor):
            activation_memory += tensor_memory(output) / (1024**3)  # Convert to GB
        elif isinstance(output, (tuple, list)):
            activation_memory += sum(tensor_memory(o) for o in output if isinstance(o, torch.Tensor)) / (1024**3)  # Convert to GB

    hooks = []
    for layer in model.modules():
        hooks.append(layer.register_forward_hook(hook_fn))

    # Run a forward pass to estimate activation memory
    with torch.no_grad():
        #sample_input = sample_input.half().to(next(model.parameters()).device)  # Ensure sample input is in float16
        #cast in float16
        with torch.autocast(device_type = "cuda", dtype = torch.float16):
            model(sample_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Total memory estimation
    total_memory = param_memory + grad_memory + activation_memory

    # Optimizer state memory estimation (approximation)
    optimizer_state_memory = 2 * param_memory  # AdamW stores two states per parameter (momentum and variance)
    total_memory += optimizer_state_memory
    print(f"Total memory: {total_memory:.2f} GB")
    print(f"Parameter memory: {param_memory:.2f} GB")
    print(f"Gradient memory: {grad_memory:.2f} GB")
    print(f"Activation memory: {activation_memory:.2f} GB")
    print(f"Optimizer state memory: {optimizer_state_memory:.2f} GB")

    return {
        'param_memory_gb': param_memory,
        'grad_memory_gb': grad_memory,
        'activation_memory_gb': activation_memory,
        'optimizer_state_memory_gb': optimizer_state_memory,
        'total_memory_gb': total_memory
    }
    





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
def print_gpu_memory_usage():
    if torch.cuda.is_available():
        # Current GPU memory usage by tensors in bytes
        memory_allocated = torch.cuda.memory_allocated()
        
        # Total GPU memory reserved by the caching allocator in bytes
        memory_reserved = torch.cuda.memory_reserved()
        
        # Convert to MB
        memory_allocated_MB = memory_allocated / (1024 ** 2)
        memory_reserved_MB = memory_reserved / (1024 ** 2)
        
        print(f"GPU memory allocated: {memory_allocated_MB:.2f} MB")
        print(f"GPU memory reserved: {memory_reserved_MB:.2f} MB")
    else:
        print("CUDA is not available on this system.")

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
        torch_dtype=torch.float16,
        device_map='cuda:0',
        cache_dir = cache_dir,
        attn_implementation = "eager",
        quantization_config = nf4_config, 
    )

    print("loading quantized model")
    model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path,cache_dir = cache_dir)
    #tinyllama pad token id is same as eos token id which is bad for finetuning because
    #either you can't ignore pad token loss as the model will not learnt to predict eos token
    #else loss will be dominated by pad token loss
    #so we set the pad token id to unk token id
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'right'
    #print("gpu usage after loading model")
    #print(print_gpu_memory_usage())
    #print(torch.cuda.memory_summary(device = "cuda"))
    #sleep 
    #time.sleep(10)


    #cascaded lora model
    print("creating cascaded lora model")
    create_cascaded_lora_model_from_quantized_model(model)
    model = set_gradient_for_all_layers(model, base_layer = False, first_adapter_layer = True)

    #print(torch.cuda.memory_summary(device = "cuda"))
    # print("gpu usage after creating cascaded lora model")
    # print(print_gpu_memory_usage())
    # time.sleep(10)
    print_trainable_parameters(model)
    print(model)
    count_parameter_bytes(model)
    print("model created")
    train_dataset = load_dataset_from_path_phi(train_dataset_path, attribute)
    if train_dataset_size != -1:
        train_dataset = train_dataset.select(range(train_dataset_size))
    test_dataset = load_dataset_from_path_phi(test_dataset_path, attribute)
    if test_dataset_size != -1:
        test_dataset = test_dataset.select(range(test_dataset_size))
        
    print("train dataset size", len(train_dataset))
    print("test dataset size", len(test_dataset))

    #remove all the columns except the text column
    train_column_names = train_dataset.column_names


    processed_train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=train_column_names,
        desc="Applying chat template to train_sft",
    )


    test_column_names = []

    processed_test_dataset = test_dataset.map(
        apply_inference_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=test_column_names,
        desc="Applying chat template to test_sft",
    )
    
    #remove all instances where len(prompt) > 2048 in the test dataset
    processed_test_dataset = processed_test_dataset.filter(lambda x: len(tokenizer(x["messages_for_inference"], return_tensors="pt")["input_ids"]) <= max_sequence_length - max_new_tokens - 10)
    print("after filtering the dataset size is : {0}".format(len(processed_test_dataset)))

    #TODO
    #1 set gradient for all layer should be made as method for the model class itself
#--------------------------------- CHECK WHETHER FREEZING IS WORKING OR NOT -------------------------------------#
    #only set the first adapter layer to be trainable
    model = set_gradient_for_all_layers(model, base_layer = False, first_adapter_layer = True)
    #     print(f"Name: {name}")
    #     print(f"  DataType: {param.dtype}")
    #     print(f"  Device: {param.device}")
    #     print(f"  Requires Grad: {param.requires_grad}")
    #     print()
#--------------------------------- \CHECK WHETHER FREEZING IS WORKING OR NOT -------------------------------------#
#------------------------------------LOOK AT THE DATASET ------------------------------------------

    #look at train_inputs 
    # print("train data ")
    # train_example = processed_train_dataset[0]
    # print(train_example['text'])
    # # print(train_example.keys())
    # print(len(train_example['input_ids']))
    # # print(len(train_example['attention_mask']))
    # # print(len(train_example['labels']))
    # # print(train_example['text'])
    # print(tokenizer.decode(train_example['input_ids'], skip_special_tokens=True))
    # print(tokenizer.decode(train_example['labels'], skip_special_tokens=False))
    # # #assert the input ids and labels 
    # # #this should be the case for causalLM
    # # # the right shift happens inside the model
    # # assert torch.equal(torch.tensor(train_example['input_ids']), torch.tensor(train_example['labels'])), "input ids and labels are not equal"
    # # print(train_example['input_ids'])
    
    
    # print("-------------------------------------")
    
    # print("test_data")
    # test_example = processed_test_dataset[0]
    # # print(test_example.keys())
    # print(test_example['messages_for_inference'])
    # print("-----------------------------------")
    
    
#----------------------------------------------------------------------------------------------#
#-----------------------CHECK FORWARD_PASS AND GENERATION------------------------------------#

    #test if generate and forward works or not
    #put the model to device first 
    #dont do model.to("cuda") as this is set to 4bit and hence automatically goes to cuda
    # print(model.device)

    # train_input_ids = torch.tensor(processed_train_dataset[0]['input_ids']).unsqueeze(0).to("cuda")
    # start_time = time.time()
    # with torch.no_grad():
    #     output = model(train_input_ids, labels = train_input_ids)
    # end_time = time.time()
    # print("time taken for forward pass : ", end_time - start_time)
    
    
    # #import code; code.interact(local = locals())

    
    # print("forward pass output loss: ")
    # #print(output)
    # #print(output.logits.shape)
    # #---notes---
    # #forward pass output is CAUSALLMOUTPUTWITHPAST object which .loss = {logits : tensor} if not labels are provided
    # #if labels are provided then loss = tensor
    # #it also output.logits which is the logits of the model
    # #FORWARD PASS IS WORKING
    # print(output.loss)
    
    
    
    # #check if generate works
    # model.eval()
    # start_time = time.time()
    # #test_input_prompt = processed_test_dataset[0]['messages_for_inference']
    # test_input_prompt = "<system> \n You are friendly chatbot</s> \n<|user|>\nhello how are you doing ?</s> \n<|assistant|>\n"
    # tokenized_test_input_prompt = tokenizer(test_input_prompt, return_tensors="pt").to("cuda")
    # print(tokenizer.decode(tokenized_test_input_prompt['input_ids'][0], skip_special_tokens = False))
    # print("-------------------------input stuff ---------------------------------------")
    # print(len(tokenized_test_input_prompt['input_ids'][0]))
    # print(tokenizer.decode(tokenized_test_input_prompt['input_ids'][0]))
    # generation_output = model.generate(**tokenized_test_input_prompt, max_new_tokens = max_new_tokens, do_sample = True, top_k = top_k, top_p = top_p, temperature = temperature, return_dict_in_generate=True, output_scores=True)
    # print("-------------------\input stuff ---------------------------------------")
    # print("-------------------output stuff ---------------------------------------")
    # #print("generation output : ", generation_output)
    # print(generation_output.sequences)
    # print(tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True))
    # end_time = time.time()
    # print("time taken for generation : ", end_time - start_time)
    # print("time taken for generation per token : ", (end_time - start_time) / len(generation_output.sequences[0]))
    # print("size of the generated sequence : ", len(generation_output.sequences[0]))
    # print(len(generation_output.scores))
    # print(generation_output.scores[0].shape)
    # print("probabilities of eos token across the sequence : ")
    # for i in range(len(generation_output.scores)):
    #     print(generation_output.scores[i][0,tokenizer.eos_token_id],end = " ")
    

#----------------------------------GENERATION AND FORWARD PASS SEEMS TO BE WORKING-----------------------------------#
#---------------------------------------------CHECK WHICH PARAMETERS BACKWARD IS UPDATING---------------------------------#
    #set the model to train mode
    #print(processed_train_dataset[0].keys()

    # Define a hook function to print layer names
# Define a hook function to print layer names and class names
    # def hook_fn(layer_name):
    #     def hook(module, input, output):
    #         print(f"Layer name in state_dict: {layer_name} | Layer class: {module.__class__.__name__}")
    #     return hook

    # # Register the hook function to each layer with its full name
    # for layer_name, module in model.named_modules():
    #     module.register_forward_hook(hook_fn(layer_name))
    #train_input_ids = torch.tensor(processed_train_dataset[0]['input_ids']).unsqueeze(0).to("cuda")
    #labels = torch.tensor(processed_train_dataset[0]['labels']).unsqueeze(0).to("cuda")
    #vram_estimate = estimate_vram_in_gb(model, train_input_ids)
    #print_device_and_dtype(model)
    #print(vram_estimate)
    
    #save the initial state dict to a file
    #torch.save(model.state_dict(), "initial_state_dict.pth")
    
    #run the forward pass with torchbf16
    #SGD optimizer
    #optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
    optimizer = configure_optimizers(model , weight_decay=0.1, learning_rate=6e-4)
    #optimizer.zero_grad()
    # from https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    #print_trainable_parameters(model)
    #model = torch.compile(model)


    #torch.autograd.set_grad_enabled(False)    
    #scaler = torch.cuda.amp.GradScaler()
    #model.gradient_checkpointing_enable()
    #model.enable_input_require_grads()
    # print(model)
    # for i in tqdm.tqdm(range(10)):
    #     with torch.autocast(device_type = "cuda", dtype = torch.float16):
    #         output = model(train_input_ids)
    #         loss = output.loss
    #         print(output)
        #scaler.scale(loss).backward()
        #scaler.step(optimizer)
        #scaler.update()
        #optimizer.zero_grad()
    #save the final state dict to a file
   # torch.save(model.state_dict(), "final_state_dict.pth")
   
    model.train()
    torch.set_float32_matmul_precision('high') # this is not the higest precision 32 : is seen as sum of 16 + 16
    #dataloader = DataLoader(processed_train_dataset, batch_size=batch_size_per_device)
    #accelerator = Accelerator(fp16=fp_16)
    
    # Convert to PyTorch tensors
    processed_train_dataset.set_format(type='torch', columns=['input_ids', 'labels'])

    # Create DataLoader
    dataloader = DataLoader(processed_train_dataset, batch_size=batch_size_per_device, shuffle=True)

    if do_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer = configure_optimizers(model , weight_decay=0.1, learning_rate=6e-4)
    
    #model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    

    #for step, batch in tqdm.tqdm(enumerate(dataloader, start=1)):
        #print(batch.keys())
        
    example = processed_train_dataset[0]
    input_ids = example["input_ids"].unsqueeze(0).to("cuda:0")
    labels = example["labels"].unsqueeze(0).to("cuda:0")
    
    
    
    #it seems that llamaforcausal does right shifting but not padding mask, lets do it ourselves,
    pad_masked_labels = input_ids.masked_fill(input_ids == tokenizer.pad_token_id, -100)
    print(pad_masked_labels)
    
    scaler = torch.cuda.amp.GradScaler()
    optimizer.zero_grad()
    
    #with torch.autocast(device_type = "cuda", dtype = torch.float16):
    output = model(input_ids, labels = pad_masked_labels)
    loss = output.loss
    #scaler.scale(loss).backward()
    #scaler.step(optimizer)
    #scaler.update()
    loss.backward()
    optimizer.step()
    
    print("done")

    
    
    
    
    
    
    
#     loss = loss / gradient_accumulation_steps
#     accelerator.backward(loss)
#     if step % gradient_accumulation_steps == 0:
#         optimizer.step()
#         optimizer.zero_grad()
# print("done")




    
