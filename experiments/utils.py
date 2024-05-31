#reference = https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing#scrollTo=a9EUEDAl0ss3

import torch 
from  config import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from dataset import create_huggingface_dataset_from_dictionary, create_multiattribute_dataset_from_dictionary
import json
from datasets import Dataset
from phi_dataset import create_huggingface_dataset_from_dictionary_for_phi , create_multiattribute_dataset_from_dictionary_for_phi



def print_trainable_parameters(model : torch.nn.Module):
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
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def get_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated() / 1024**3  # Convert bytes to GB
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # Convert bytes to GB
    return allocated, max_allocated

def load_dataset_from_path(dataset_path : str, controllable_aspect : str) -> Dataset:
    dataset_dict = json.load(open(dataset_path,"r"))
    dataset = create_huggingface_dataset_from_dictionary(dataset_dict, controllable_aspect)
    return dataset

def load_multi_attribute_dataset_from_path(dataset_path : str, controllable_aspects : list) -> Dataset:
    dataset_dict = json.load(open(dataset_path,"r"))
    dataset = create_multiattribute_dataset_from_dictionary(dataset_dict, controllable_aspects)
    return dataset

def load_dataset_from_path_phi(dataset_path : str, controllable_aspect : str) -> Dataset:
    dataset_dict = json.load(open(dataset_path,"r"))
    dataset = create_huggingface_dataset_from_dictionary_for_phi(dataset_dict, controllable_aspect)
    return dataset

def load_multi_attribute_dataset_from_path(dataset_path : str, controllable_aspects : list) -> Dataset:
    dataset_dict = json.load(open(dataset_path,"r"))
    dataset = create_multiattribute_dataset_from_dictionary_for_phi(dataset_dict, controllable_aspects)
    return dataset

def formatting_prompts_func(examples : dict) -> list:
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        response = examples["output"][i]

        text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        
        ### Instruction:
        {instruction}
        
        ### Input:
        {input_text}
        
        ### Response:
        {response}
        '''

        output_text.append(text)
    return output_text






#def run_inference_model(model, dataset)

def count_model_parameters(model : torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__== "__main__":
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = True)
    one_million = 10**6
    print(f"Model name: {model_name}")
    print(f"Model parameters: {count_model_parameters(model)/one_million} M")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU device found")
    else:
        print("GPU device not found")
        exit()

    allocated, max_allocated = get_gpu_memory_usage()
    print(f"Current GPU memory usage: {allocated:.2f} GB")
    print(f"Peak GPU memory usage: {max_allocated:.2f} GB")
    


