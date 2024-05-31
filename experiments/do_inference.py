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
from eval import output_metrics

# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)

def apply_inference_chat_template(
        example, 
        tokenizer,
    ):
    
    messages = example["messages"]
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system",
                            "content": "You are a friendly chatbot who always responds in the style of a pirate"
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
                            "content": "You are a friendly chatbot who always responds in the style of a pirate"
                                })
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False)
    return example
if __name__ == "__main__":
        #parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_directory", type=str, help="model directory", default = "/scratch/tathagato/adapter_experiments/2024-05-18-03-29-07_length_length_macsum_TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--checkpoint_path", type=str, help="model name", default = "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--tokenizer_name", type=str, help="tokenizer name", default = "NousResearch/Llama-2-7b-hf")
    parser.add_argument("--train_dataset_path", type=str, help="train dataset path", default = "/home2/tathagato/summarization/MACSum/dataset/macdoc/train_dataset.json")
    parser.add_argument("--test_dataset_path", type=str, help="test dataset path", default = "/home2/tathagato/summarization/MACSum/dataset/macdoc/test_dataset.json")
    parser.add_argument("--val_dataset_path", type=str, help="val dataset path", default = "/home2/tathagato/summarization/MACSum/dataset/macdoc/val_dataset.json")
    parser.add_argument("--test_dataset_size", type=int, help="test dataset size", default = -1)
    parser.add_argument("--train_dataset_size", type=int, help="train dataset size", default = -1)
    parser.add_argument("--val_dataset_size", type=int, help="val dataset size", default = -1)
    parser.add_argument("--cache_dir", type=str, help="cache directory", default = "/scratch/tathagato")
    
    parser.add_argument("--instruction_template", type=str, help="instruction template", default = "### Instruction:")
    parser.add_argument("--response_template", type=str, help="response template", default = "### Response:")
    
    #lora config
    parser.add_argument("--r", type=int, help="rank", default = 32)
    parser.add_argument("--lora_alpha", type=int, help="learning rate", default = 16)
    parser.add_argument("--lora_dropout", type=float, help="dropout", default = 0.1)
    parser.add_argument("--bias", type=str, help="bias", default = "none")
    parser.add_argument("--task_type", type=str, help="task type", default = "CAUSAL_LM")
    parser.add_argument("--target_modules", type=list, help="target modules", default = ["qkv_proj"])

    
    #nf4 config
    parser.add_argument("--load_in_4bit", type=bool, help="load in 4 bit", default = True)
    parser.add_argument("--bnb_4bit_quant_type", type=str, help="bnb 4 bit quant type", default = "nf4")
    parser.add_argument("--bnb_4bit_use_double_quant", type=bool, help="bnb 4 bit use double quant", default = True)
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, help="bnb 4 bit compute dtype", default = "torch.float16")

    


    #training args
    parser.add_argument("--output_dir", type=str, help="output directory", default = "/scratch/tathagato/zero_shot_inference_experiments")
    parser.add_argument("--num_train_epochs", type=int, help="number of training epochs", default = 10)
    parser.add_argument("--per_device_train_batch_size", type=int, help="batch size", default = 1)
    parser.add_argument("--gradient_accumulation_steps", type=int, help="gradient accumulation steps", default = 2)
    parser.add_argument("--gradient_checkpointing", type=bool, help="gradient checkpointing", default = True)
    parser.add_argument("--optim", type=str, help="optimizer", default = "adamw_bnb_8bit")
    parser.add_argument("--learning_rate", type=float, help="learning rate", default = 5e-5)
    parser.add_argument("--warmup_ratio", type=int, help="warmup steps", default = 0.03)
    parser.add_argument("--max_grad_norm", type=int, help="max grad norm", default = 0.3)
    parser.add_argument("--seed", type=int, help="seed", default = 42)
    parser.add_argument("--lr_scheduler_type", type=str, help="lr scheduler type", default = "constant")
    parser.add_argument("--report_to", type=str, help="report to", default = "wandb")
    parser.add_argument("--attribute", type=str, help="attribute", default = "length")
    parser.add_argument("--load_previous_model", action= "store_true", help="load previous model")
    parser.add_argument("--previous_model_path", type=str, help="previous model path", default = "/scratch/tathagato/openelm_adapter experiments/2022-02-22-14-47-00_length")
    parser.add_argument("--use_current_adapter",  type=bool, help="use current adapter", default = True)
    parser.add_argument("--run_inference", action= "store_true", help="run inference")
    parser.add_argument("--use_checkpoint", action = "store_true", help="use checkpoint")
    parser.add_argument("--output_file", type=str, help="output file", default = "./results/output.json")
    parser.add_argument("--use_2_checkpoint", action = "store_true", help="use 2 checkpoint") 
    parser.add_argument("--first_checkpoint_path", type=str, help="first checkpoint path", default = "/scratch/tathagato/adapter_experiments/2024-05-18-03-29-07_length_length_macsum_TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--use_merged_model_checkpoint", action = "store_true", help="use merged model checkpoint", default= False)
    parser.add_argument("--merged_model_directory", type = str, help="merged model directory", default= "/scratch/tathagato/adapter_experiments/2024-05-18-03-29-07_length_length_macsum_TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--first_adapter_name", type=str, help="first adapter name", default = "length")
    parser.add_argument("--second_adapter_name", type=str, help="second adapter name", default = "length")
    parser.add_argument("--second_checkpoint_path", type=str, help="second checkpoint path", default = "/scratch/tathagato/adapter_experiments/2024-05-18-03-29-07_length_length_macsum_TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args = parser.parse_args()
    #get the current date and time in human format
    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_" + str(args.attribute)
    #set the seed
    seed = args.seed
    torch.manual_seed(seed)

    # else:
    #     args.output_dir = os.path.join(args.output_dir,args.attribute +  current_date)
    logger = logging.getLogger(__name__)




    nf4_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=torch.float16
    )






    if args.use_checkpoint:
        print("loading from " + args.checkpoint_path)
        tokenizer_path = os.path.dirname(args.model_directory)
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path,
                                            trust_remote_code = True, 
                                            quantization_config = nf4_config, 
                                            torch_dtype = torch.float16,
                                            device_map = "cuda:0")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code = True)
        config = PeftConfig.from_pretrained(args.model_directory)
        model = PeftModelForCausalLM.from_pretrained(model, args.model_directory, args.first_adapter_name, is_trainable = True)
        print(list(model.peft_config.keys()))
        model = model.merge_and_unload()
        tokenizer.padding_side = 'left'
    elif args.use_merged_model_checkpoint:
        print("loading from " + args.merged_model_directory)
        merged_model_checkpoint_path = os.path.join(args.merged_model_directory, "final_merged_model")
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path,
                                            trust_remote_code = True, 
                                            quantization_config = nf4_config, 
                                            torch_dtype = torch.float16,
                                            device_map = "cuda:0")

        tokenizer = AutoTokenizer.from_pretrained(args.merged_model_directory, trust_remote_code = True)
        #config = PeftConfig.from_pretrained(args.merged_model_directory)
        tokenizer.padding_side = 'left'
    elif args.use_2_checkpoint:
        print("two checkpoint loading")
        print("adapter_1_path : {0}".format(args.first_checkpoint_path))
        print("adapter_2_path : {0}".format(args.second_checkpoint_path))
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path,
                                            trust_remote_code = True, 
                                            quantization_config = nf4_config, 
                                            torch_dtype = torch.float16,
                                            device_map = "cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir = args.cache_dir)
        model = PeftModelForCausalLM.from_pretrained(model, args.first_checkpoint_path, args.first_adapter_name)
        print(list(model.peft_config.keys()))
        #model.load_adapter(args.second_checkpoint_path, args.second_adapter_name)
        model = PeftModelForCausalLM.from_pretrained(model, args.second_checkpoint_path, args.second_adapter_name)
        print(list(model.peft_config.keys()))

        model = model.merge_and_unload()
        tokenizer.padding_side = 'left'


    else:
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, quantization_config = nf4_config, torch_dtype = torch.float16, device_map = "cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir = args.cache_dir)
        tokenizer.padding_side = 'left'


    print("attribute for the dataset is : {0}".format(args.attribute))
    test_dataset = load_dataset_from_path_phi(args.val_dataset_path, args.attribute)
    if args.test_dataset_size == -1:
        args.test_dataset_size = len(test_dataset)

    # test_dataset = test_dataset.select(range(10))


    print("test dataset size", len(test_dataset))


    column_names = []

    processed_test_dataset = test_dataset.map(
        apply_inference_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to test_sft",
    )
    print("tokenizer padding side", tokenizer.padding_side)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
 

    #running inference using pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, batch_size = 4)


    # # Initialize variables
    prompts = []
    batch_size = 4 # Adjust the batch size as needed
    max_sequence_length = 2048  # Set this to your model's maximum sequence length
    dist = {}
    all_outputs = []
    total_input_size = 0
    inference_output = {}
    cur_index = 0
    index_matches = []
    for index, example in tqdm.tqdm(enumerate(processed_test_dataset)):
        messages = example["messages"]
        messages[-1]["content"] = ""
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Check if the tokenized prompt length exceeds the maximum sequence length
        tokenized_prompt = pipe.tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_length = tokenized_prompt.size(1)
        if prompt_length not in dist:
            dist[prompt_length] = 0
        dist[prompt_length] += 1
        #print(f"Example {index} has length {prompt_length}")
        if tokenized_prompt.size(1) > max_sequence_length:
            #print(f"Skipping example {index} due to length {tokenized_prompt.size(1)} > {max_sequence_length}")
            total_input_size += 1
            continue  # Skip this example if it exceeds the maximum sequence length

        prompts.append(prompt)
        index_matches.append(cur_index)
        inference_output[cur_index] = {key : example[key] for key in example.keys()}
        inference_output[cur_index]['prompt_for_inference'] = prompt
        cur_index += 1


        # Check if we have reached the batch size
        if len(prompts) == batch_size:
            # Process the batch
            outputs = pipe(prompts, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            size = len(outputs)
            for i in range(size):
                inference_output[index_matches[i]]['generated_text'] = outputs[i][0]['generated_text']


            # Reset the prompts list for the next batch
            index_matches = []
            prompts = []

    #dump the output 
    # if args.use_checkpoint:
    #     output_file_path = os.path.join(args.model_directory, f"{args.attribute}_output.json")
    #     print("saving the output file : {0}".format(output_file_path))
    #     with open(output_file_path, "w") as f:
    #         json.dump(inference_output, f)
    # else:
    #     output_file_path = os.path.join(args.output_dir, f"{args.attribute}_output.json")
    #     print("saving the output file : {0}".format(output_file_path))
    #     with open(output_file_path, "w") as f:
    #         json.dump(inference_output, f)

    # print("The size of the dataset is: ", len(processed_test_dataset))

    print("The size of the dataset is: ", len(processed_test_dataset))
    print("Saving the output file to {0}".format(args.output_file))
    print("length of the inference output", len(inference_output))
    with open(args.output_file, "w") as f:
        json.dump(inference_output, f)
    print("Saved the model output to {0}".format(args.output_file))

    #output 
    output_metrics(args.output_file, args.attribute)






        







