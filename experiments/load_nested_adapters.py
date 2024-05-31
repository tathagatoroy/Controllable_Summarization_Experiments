import sys
import logging
import tqdm
import datasets
from datasets import load_dataset
from peft import LoraConfig
import torch
import transformers
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from phi_dataset import create_huggingface_dataset_from_dictionary_for_phi , create_multiattribute_dataset_from_dictionary_for_phi
import argparse
from datetime import datetime
import os
from utils import load_dataset_from_path_phi, load_multi_attribute_dataset_from_path,print_trainable_parameters, count_model_parameters
from transformers import pipeline
from peft import PeftModelForCausalLM, get_peft_config, PeftConfig
import json

def compare_model_weights(model_a, model_b):
    
    """
    Compares the weights of two PyTorch models.

    Args:
        model_a: First model to compare.
        model_b: Second model to compare.

    Returns:
        True if all weights are identical, False otherwise.
    """
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()
    same = True
    for name, param_a in state_dict_a.items():
        if name not in state_dict_b:
            print(f"Missing parameter {name} in model_b")
            same = False
        else:
            if not torch.allclose(param_a, state_dict_b[name]):
                print(f"Parameter {name} does not match")
                same = False
    for key in state_dict_b.keys():
        print(key)
        
    print("\n\n------------------------------\n\n")
    for key in state_dict_a.keys():
        print(key)
    return same


if __name__=="__main__":
    #dual checkpoint 
    checkpoint_folder = ["/scratch/tathagato/new_adapter_experiments/length","/scratch/tathagato/new_adapter_experiments/not_sure/length_then_topic/topic"]
    nf4_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_use_double_quant=True,
       bnb_4bit_compute_dtype=torch.float16


    )
    print("loading first model")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                                            trust_remote_code = True, 
                                            quantization_config = nf4_config, 
                                            torch_dtype = torch.float16,
                                            device_map = "cuda:0")



    model = PeftModelForCausalLM.from_pretrained(model,"/scratch/tathagato/new_adapter_experiments/length/", "length")
    print(list(model.peft_config.keys()))
    print(model)


    model.load_adapter("/scratch/tathagato/new_adapter_experiments/not_sure/length_then_topic/topic", "topic")
    print(list(model.peft_config.keys()))
    print(model)

    model = model.merge_and_unload()
    print(list(model.peft_config.keys()))
    print(model)
                                    

