from config import *
from accelerate import init_empty_weights, infer_auto_device_map
from transformers import AutoConfig , AutoModelForCausalLM

if __name__ == "__main__":
    config = AutoConfig.from_pretrained(model_name)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    device_map = infer_auto_device_map(model)
    print(device_map)
