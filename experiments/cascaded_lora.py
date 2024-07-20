import sys
import logging
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit
import datasets
from trl import SFTTrainer
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, set_seed
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
import time
import numpy as np
import random
###################################################################################################################################################################
#----------------------------------------------------------------TODOS----------------------------------------------------------------------------------------------#
''' 
1.check there still exist some discrepancy between train and test prompt. That is there should </s> after each role prompt but <s> only in the beginning of the prompt 
that is currently not the case ---> this looks fine I think. FIXED
2. check in generate if model is giving the same output without using the 2 cascading layers. It should be same as base model. ---> IT APPEARS TO BE SAME.
3. check which layers weight change if called loss.backward 

'''
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------#
###################################################################################################################################################################
#config 
rank_1 = 64
rank_2 = 32
alpha_1 = 16
alpha_2 = 16
adapter_name = "default"
dropout = 0.1
is_second_layar_being_trained = False
is_first_layer_being_trained = False
is_first_layer_being_used_for_inference = False
is_second_layer_being_used_for_inference = False

target_modules = [
                    'q_proj',
                    'k_proj',
                    'v_proj',
                    'o_proj',
                    'gate_proj',
                    'up_proj',
                    'down_proj'
]
cache_dir = "/scratch/murali"
#generation config 
max_new_tokens = 150
top_k = 50
top_p = 0.95
temperature = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#dataset config 
attribute = "length"
train_dataset_path = "../dataset/macdoc/train_dataset.json"
test_dataset_path = "../dataset/macdoc/test_dataset.json"
test_dataset_size = 10 #-1 for all
train_dataset_size = 10 #-1 means for all

#model config 
max_seq_length = 2048
max_sequence_length = 2048

#training config
learning_rate = 3e-4
num_train_epochs = 1
batch_size_per_device = 1
num_device = 4
gradient_accumulation_steps = 2




#taken from karpathy. Not needed for now as generate works
# def generate(model, tokenizer, prompt , max_new_tokens = max_new_tokens, top_k = top_k, top_p = top_p, temperature = temperature, device = device):
#     model.eval()
#     num_return_sequences = 1
#     max_length = max_new_tokens
#     tokens = tokenizer.encode(prompt)
#     tokens = torch.tensor(tokens, dtype=torch.long)
#     mum_tokens_intially = tokens.size(0)
#     tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
#     xgen = tokens.to(device)
#     sample_rng = torch.Generator(device=device)
#     sample_rng.manual_seed(42)
#     while xgen.size(1) - num_tokens_initially < max_length:
#         # forward the model to get the logits
#         with torch.no_grad():
#             with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
#                 logits, loss = model(xgen) # (B, T, vocab_size)
#             # take the logits at the last position
#             logits = logits[:, -1, :] # (B, vocab_size)
#             # get the probabilities
#             probs = F.softmax(logits, dim=-1)
#             # do top-k sampling of 50 (huggingface pipeline default)
#             # topk_probs here becomes (5, 50), topk_indices is (5, 50)
#             topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#             # select a token from the top-k probabilities
#             # note: multinomial does not demand the input to sum to 1
#             ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
#             # gather the corresponding indices
#             xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
#             # append to the sequence
#             xgen = torch.cat((xgen, xcol), dim=1)
#     # print the generated text
#     for i in range(num_return_sequences):
#         tokens = xgen[i, :max_length].tolist()
#         decoded = enc.decode(tokens)
#         print(f"rank {ddp_rank} sample {i}: {decoded}")
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
        self.is_second_layar_being_trained = is_second_layar_being_trained
        self.is_first_layer_being_trained = is_first_layer_being_trained
        self.is_first_layer_being_used_for_inference = is_first_layer_being_used_for_inference
        self.is_second_layer_being_used_for_inference = is_second_layer_being_used_for_inference
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
        output = None
        if self.is_first_layer_being_used_for_inference and self.is_second_layer_being_used_for_inference:
            #print("first and second both")
            #x = self.scaling_1 * (x @ self.W1_a @ self.W1_b) + self.scaling_2 * (x @ self.W2_a1 @ self.W2_a2 @ self.W2_b1 @ self.W2_b2)
            output  = self.base_layer(x) + self.scaling_1 * (self.lora_A[self.adapter_name]) + self.scaling_2 * (self.W2['B2'](self.W2['A2'](self.W2['B1'](self.W2['A1'](x)))))
        elif self.is_first_layer_being_used_for_inference and not self.is_second_layer_being_used_for_inference:
            #x = self.scaling_2 * (x @ self.W2_a1 @ self.W2_a2) 
            #print("first only")
            output  =  self.base_layer(x)  + self.scaling_1 * (self.W1['A'](self.W1['B'](x))) 

        elif not self.is_first_layer_being_used_for_inference and self.is_second_layer_being_used_for_inference:
            #x = self.scaling_1 * (x @ self.W1_a @ self.W1_b) 
            #print("second only")
            output  = self.base_layer(x) + self.scaling_2 * (self.W2['B2'](self.W2['A2'](self.W2['B1'](self.W2['A1'](x)))))
        else:
            #print("none")
            output = self.base_layer(x)
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
def compare_state_dicts(initial_dict, final_dict):
    modified_layers = []
    for layer_name in initial_dict:
        if not torch.equal(initial_dict[layer_name], final_dict[layer_name]):
            modified_layers.append(layer_name)
    return modified_layers

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

if __name__ == "__main__":
    #set seed
    set_all_seeds(42)

    
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
        cache_dir = cache_dir,
        attn_implementation = "eager",
        quantization_config = nf4_config, 
    )
    print("loading quantized model")
    base_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path,cache_dir = cache_dir)
    #tinyllama pad token id is same as eos token id which is bad for finetuning because
    #either you can't ignore pad token loss as the model will not learnt to predict eos token
    #else loss will be dominated by pad token loss
    #so we set the pad token id to unk token id
    tokenizer.pad_token = tokenizer.unk_token


    #cascaded lora model
    print("creating cascaded lora model")
    #create_cascaded_lora_model_from_quantized_model(model)
    print("model created")
    tokenizer.padding_side = 'right'
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

    
    #initial_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

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
    # model.train()
    # train_input_ids = torch.tensor(processed_train_dataset[0]['input_ids']).unsqueeze(0).to("cuda")
    # labels = torch.tensor(processed_train_dataset[0]['labels']).unsqueeze(0).to("cuda")
    
    # #save the initial state dict to a file
    # torch.save(model.state_dict(), "initial_state_dict.pth")
    
    # #run the forward pass with torchbf16
    # optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
    # optimizer.zero_grad()
    # # from https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    # torch.set_float32_matmul_precision('high') # this is not the higest precision 32 : is seen as sum of 16 + 16
    
    # scaler = torch.cuda.amp.GradScaler()
    # for i in tqdm.tqdm(range(10)):
    #     with torch.autocast(device_type = "cuda", dtype = torch.float16):
    #         output = model(train_input_ids, labels = labels)
    #         loss = output.loss
    #     scaler.scale(loss).backward()
    #     scaler.step(optimizer)
    #     scaler.update()
    #     optimizer.zero_grad()
    # #save the final state dict to a file
    # torch.save(model.state_dict(), "final_state_dict.pth")

    # for epoch in range(0): # 0 epochs, this section is for illustration only
    #     for input, target in zip(data, targets):
    #         with torch.autocast(device_type=device, dtype=torch.float16):
    #             output = net(input)
    #             loss = loss_fn(output, target)

    #         # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
    #         scaler.scale(loss).backward()

    #         # ``scaler.step()`` first unscales the gradients of the optimizer's assigned parameters.
    #         # If these gradients do not contain ``inf``s or ``NaN``s, optimizer.step() is then called,
    #         # otherwise, optimizer.step() is skipped.
    #         scaler.step(opt)

    #         # Updates the scale for next iteration.
    #         scaler.update()

    #         opt.zero_grad() # set_to_none=True here can modestly improve performance




    





    # Assuming `model` is your neural network model
    for name, param in model.named_parameters():
        print(f"Name: {name}")
        print(f"  DataType: {param.dtype}")
        print(f"  Device: {param.device}")
        print(f"  Requires Grad: {param.requires_grad}")
        print()