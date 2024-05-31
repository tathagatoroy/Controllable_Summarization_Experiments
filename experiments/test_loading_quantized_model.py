import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb

if __name__ == "__main__":

    model_name = "/scratch/tathagato/adapter_experiments/extractiveness/final_merged_model"

    # Load the tokenizer
    #tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the model with quantization
    model = AutoModelForCausalLM.from_pretrained(model_name,  device_map="auto")

    # # Example to inspect quantization specific configuration
    # if hasattr(model, 'quantization_config'):
    #     quantization_config = model.quantization_config
    #     print("Quantization Configuration:")
    #     print(quantization_config)
    # else:
    #     print("Quantization config not found in the model attributes.")
    for attr in dir(model):
        # Filter out methods and built-in attributes
        if not callable(getattr(model, attr)) and not attr.startswith("__"):
            print(f"{attr}: {getattr(model, attr)}")


