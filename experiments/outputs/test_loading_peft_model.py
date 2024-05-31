from config import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModelForCausalLM, get_peft_config, PeftConfig
if __name__=="__main__":
    
    checkpoint_dir = "/scratch/tathagato/openelm_adapter experiments/2024-05-01-18-26-09_length/checkpoint-4278"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir,load_in_4bit_mode=True)
    config = PeftConfig.from_pretrained(checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModelForCausalLM.from_pretrained(model, checkpoint_dir, is_trainable = True)

    model.print_trainable_parameters()
    print(model.peft_config)

    