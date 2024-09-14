import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from bitsandbytes.nn import Linear4bit
import sys
sys.path.append("./scripts")
from hlora import HLORAConfig, replace_linear4bit_with_hlora, set_train_adapters, set_inference_adapters, set_gradients_on_the_model, print_model_layer_info, replace_lora_with_hlora
from dataset import MACSUM
from torch.nn.utils.rnn import pad_sequence
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import json 
import argparse
import tqdm
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TOKENIZERS_PARALLELISM'] = "false"
import pickle as pkl
import copy
import safetensors


def setup_hlora_model(model_id: str, lora_config: LoraConfig, hlora_config: HLORAConfig, bnb_config: BitsAndBytesConfig):
    # 1. Load the model
    model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config = bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 2. Get PEFT model
    peft_model = get_peft_model(model, lora_config)


    # 3. Replace lora layers with HLORA
    peft_model = replace_lora_with_hlora(peft_model, hlora_config)
    prepare_model_for_kbit_training(peft_model)

    return peft_model, tokenizer

def load_state_dict_from_checkpoint(model, checkpoint_path):
    #load the safetensor 
    checkpoint = safetensors.safe_open(checkpoint_path,"pt", "cuda")
    device = model.device
    for name, param in checkpoint.items():
        model.state_dict()[name].copy_(param.to(device))
    return model