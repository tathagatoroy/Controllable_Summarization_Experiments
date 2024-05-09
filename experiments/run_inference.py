#using a config to file to test it out for now , will parameterize the code later
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, get_kbit_device_map
from utils import print_trainable_parameters, formatting_prompts_func, load_dataset_from_path, count_model_parameters, load_multi_attribute_dataset_from_path
from transformers import TrainingArguments
from accelerate import Accelerator
import argparse
from datetime import datetime
import wandb
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from transformers import TrainingArguments
from peft import PeftModelForCausalLM, get_peft_config, PeftConfig
import pprint
import tqdm
pretty_print = pprint.PrettyPrinter(indent=4)
import time
import pickle as pkl



if __name__ == "__main__":


    #parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="model name", default = "apple/OpenELM-450M")
    parser.add_argument("--tokenizer_name", type=str, help="tokenizer name", default = "NousResearch/Llama-2-7b-hf")
    parser.add_argument("--test_dataset_path", type=str, help="test dataset path", default = "/home2/tathagato/summarization/MACSum/dataset/macdoc/test_dataset.json")
    parser.add_argument("--val_dataset_path", type=str, help="val dataset path", default = "/home2/tathagato/summarization/MACSum/dataset/macdoc/val_dataset.json")
    parser.add_argument("--cache_dir", type=str, help="cache directory", default = "/scratch/tathagato")
    
    parser.add_argument("--instruction_template", type=str, help="instruction template", default = "### Instruction:")
    parser.add_argument("--response_template", type=str, help="response template", default = "### Response:")
    


    
    #nf4 config
    parser.add_argument("--load_in_4bit", type=bool, help="load in 4 bit", default = True)
    parser.add_argument("--bnb_4bit_quant_type", type=str, help="bnb 4 bit quant type", default = "nf4")
    parser.add_argument("--bnb_4bit_use_double_quant", type=bool, help="bnb 4 bit use double quant", default = True)
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, help="bnb 4 bit compute dtype", default = "torch.bfloat16")



    parser.add_argument("--output_dir", type=str, help="output directory", default = "/scratch/tathagato/openelm_adapter_experiments")
    parser.add_argument("--model_directory", type=str, help="model directory", default = "/scratch/tathagato/openelm_adapter_experiments/2024-05-01-18-26-09_length/checkpoint-4278")
    parser.add_argument("--use_checkpoint", action = "store_true", help="use checkpoint", default = True)
    parser.add_argument("--attributes", nargs = "+", help="attributes", type = str)
    

    args = parser.parse_args()
    #print attributes
    print("The attributes are: ", args.attributes)

    

    args.output_dir = args.model_directory


    test_dataset_path = args.test_dataset_path
    val_dataset_path = args.val_dataset_path
    cache_dir = args.cache_dir
    instruction_template = args.instruction_template
    response_template = args.response_template
    
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    
    if args.use_checkpoint:
        print("loading from " + args.model_directory)
        model = AutoModelForCausalLM.from_pretrained(args.model_directory, 
                                            trust_remote_code = True, 
                                            quantization_config = nf4_config, 
                                            torch_dtype = torch.float16,
                                            device_map = "cuda:0")

        tokenizer = AutoTokenizer.from_pretrained(args.model_directory, trust_remote_code = True)
        config = PeftConfig.from_pretrained(args.model_directory)
        model = PeftModelForCausalLM.from_pretrained(model, args.model_directory, is_trainable = True)
        adapter_name = list(model.peft_config.keys())[0]
        model.set_adapter(adapter_name)
        print("setting adapter: ", adapter_name)
        ca_aspect = "_and_".join(args.attributes)
        output_file_path = os.path.join(args.model_directory, f"{ca_aspect}_output.json")

    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                            trust_remote_code = True, 
                                            quantization_config = nf4_config, 
                                            torch_dtype = torch.float16,
                                            device_map = "cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code = True)
        ca_aspect = "_and_".join(args.attributes)

        output_file_path = os.path.join("/home2/tathagato/summarization/MACSum/experiments/results", f"{ca_aspect}_output.json")

    

    dataset = load_multi_attribute_dataset_from_path(val_dataset_path, args.attributes)
    print("The size of the dataset is: ", len(dataset))
    output_file = {}
    model.eval()
    device = torch.device("cuda:0")
    model = model.to(device)

    for index, example in tqdm.tqdm(enumerate(dataset)):
        new_example = {key : example[key] for key in example.keys()}
        generation_prompt = example['prompt_for_inference']

        with torch.no_grad():
            input_ids = tokenizer(generation_prompt, return_tensors = "pt", padding = "max_length").input_ids
            input_ids = input_ids.to(device)
            print(input_ids.shape)
            output_ids = model.generate(
                input_ids,
                max_new_tokens = 300,
                pad_token_id = 0
            )


            output_text = tokenizer.decode(
                output_ids[0].tolist(),
                skip_special_tokens=True
             )
            
            new_example['generated_text'] = output_text
            output_file[index] = new_example
            print("The prompt for generation is: ", generation_prompt)
            print("The generated text is: ", output_text)
        if index == 2:
            break
    

    #save the output file
    print("saving the output file : {0}".format(output_file_path))
    with open(output_file_path, "w") as f:
        json.dump(output_file, f)




        
    





