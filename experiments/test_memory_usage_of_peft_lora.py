import sys
import logging
import tqdm
import datasets
from datasets import load_dataset
from peft import LoraConfig,PeftConfig, PeftModel, PeftModelForCausalLM
import torch
import transformers
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig,TrainerCallback
import argparse
from datetime import datetime
import os
from utils import load_dataset_from_path_phi, load_multi_attribute_dataset_from_path,print_trainable_parameters, count_model_parameters
from transformers import pipeline
import json

# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)




def inference(model, tokenizer, processed_test_dataset, batch_size = 4):
        #running inference using pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, batch_size = 4)
    prompts = []
    max_sequence_length = 2048  # Set this to your model's maximum sequence length
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
        print(f"Example {index} has length {prompt_length}")
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
            outputs = pipe(prompts, max_new_tokens=256, do_sample=True, temperature=1, top_k=50, top_p=0.95)
            size = len(outputs)
            for i in range(size):
                inference_output[index_matches[i]]['generated_text'] = outputs[i][0]['generated_text']


            # Reset the prompts list for the next batch
            index_matches = []
            prompts = []
        return inference_output
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
class InferenceCallback(TrainerCallback):
    def __init__(self, model, tokenizer, data, inference_directory, args):
        self.model = model
        self.tokenizer = tokenizer
        self.data = data
        self.inference_directory = inference_directory
        self.args = args

    def on_save(self, args, state, control, **kwargs):
        # Perform inference
        outputs = inference(model, tokenizer, test_dataset, batch_size = 4)

        # Save results with the current save step as part of the filename
        output_file = os.path.join(self.inference_directory, f'{self.args.experiment_name}_evaluate_on_{self.args.attribute}_{state.global_step}.json')
        #make inference directory if it does not exist
        os.makedirs(self.inference_directory, exist_ok=True)
        with open(output_file, 'w') as file:
            json.dump(outputs, file)
        
        # Log the save step
        print(f'Saved inference results to {output_file} at step {state.global_step}')
        
        return control
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
    example["input_ids"] = tokenized_example["input_ids"]
    example["attention_mask"] = tokenized_example["attention_mask"]
    example['labels'] = tokenized_example["input_ids"]
    return example
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
    
    
    
if __name__ == "__main__":
        #parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, help="model name", default = "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--tokenizer_name", type=str, help="tokenizer name", default = "NousResearch/Llama-2-7b-hf")
    parser.add_argument("--train_dataset_path", type=str, help="train dataset path", default = "/home2/tathagato/summarization/MACSum/dataset/macdoc/train_dataset.json")
    parser.add_argument("--test_dataset_path", type=str, help="test dataset path", default = "/home2/tathagato/summarization/MACSum/dataset/macdoc/test_dataset.json")
    parser.add_argument("--val_dataset_path", type=str, help="val dataset path", default = "/home2/tathagato/summarization/MACSum/dataset/macdoc/val_dataset.json")
    parser.add_argument("--test_dataset_size", type=int, help="test dataset size", default = -1)
    parser.add_argument("--train_dataset_size", type=int, help="train dataset size", default = -1)
    parser.add_argument("--val_dataset_size", type=int, help="val dataset size", default = -1)
    parser.add_argument("--cache_dir", type=str, help="cache directory", default = "/scratch/tathagato")
    parser.add_argument("--inference_directory", type=str, help="inference directory", default = "./inference_results")
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
    parser.add_argument("--output_dir", type=str, help="output directory", default = "/scratch/tathagato/redo_adapter_experiments")
    parser.add_argument("--num_train_epochs", type=int, help="number of training epochs", default = 10)
    parser.add_argument("--per_device_train_batch_size", type=int, help="batch size", default = 1)
    parser.add_argument("--gradient_accumulation_steps", type=int, help="gradient accumulation steps", default = 2)
    parser.add_argument("--gradient_checkpointing", type=bool, help="gradient checkpointing", default = True)
    parser.add_argument("--optim", type=str, help="optimizer", default = "adamw_bnb_8bit")
    parser.add_argument("--learning_rate", type=float, help="learning rate", default = 6e-4)
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
    args = parser.parse_args()
    #get the current date and time in human format
    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_" + str(args.attribute)
    #set the seed
    seed = args.seed
    torch.manual_seed(seed)
    #args.output_dir = os.path.join(args.output_dir,args.attribute +  current_date)
    logger = logging.getLogger(__name__)
    if args.load_previous_model:
        base_dir_attribute = os.path.basename(args.previous_model_path)
        args.experiment_name = f"{base_dir_attribute}_then_{args.attribute}"
        args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    else:
        args.experiment_name = f"{args.attribute}"
        args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    print("output directory", args.output_dir)
    print("experiment name", args.experiment_name)
    args.output_dir = args.output_dir + "_" + str(args.learning_rate)
    args.output_dir = "/scratch/tathagato/test"
    os.makedirs(args.output_dir, exist_ok=True)


    ###################
    # Hyper-parameters
    ###################
    training_config = {
        "bf16": False,
        "fp16" : True,
        "do_eval": False,
        "learning_rate": args.learning_rate,
        "log_level": "info",
        "logging_steps": 20,
        "logging_strategy": "steps",
        "lr_scheduler_type": "cosine",
        "num_train_epochs": 8,
        "max_steps": -1,
        "output_dir": args.output_dir,
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
        "neftune_noise_alpha":5,
        }

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=torch.float16
    )

    peft_config = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": [ 'k_proj', 'o_proj', 'q_proj', 'v_proj'],
        "modules_to_save": None,
    }
    train_conf = TrainingArguments(**training_config)
    peft_conf = LoraConfig(**peft_config)
    #4278/554



    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = train_conf.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}"
        + f" distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_conf}")
    logger.info(f"PEFT parameters {peft_conf}")


    ################
    # Modle Loading
    ################
    #checkpoint_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # checkpoint_path = "microsoft/Phi-3-mini-128k-instruct"
    #use flash attention 1
    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None,
        cache_dir = "/scratch/tathagato",
        attn_implementation = "eager",
        quantization_config = nf4_config, 

    )
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path,cache_dir = "/scratch/tathagato")
    #tinyllama pad token id is same as eos token id which is bad for finetuning because
    #either you can't ignore pad token loss as the model will not learnt to predict eos token
    #else loss will be dominated by pad token loss
    #so we set the pad token id to unk token id
    tokenizer.pad_token = tokenizer.unk_token
    if args.load_previous_model:
        print("loading model from : {0}".format(args.previous_model_path))
        model = PeftModelForCausalLM.from_pretrained(model, args.previous_model_path, is_trainable = True)
        adapter_name = list(model.peft_config.keys())
        #print("adapter name", adapter_name)
        model =  model.merge_and_unload()




        #print("active2\n", model.active_adapters())
        #print("list2\n", list(model.peft_config.keys()))
    model = PeftModelForCausalLM(model, peft_config = peft_conf,adapter_name = args.attribute)
    model.set_adapter(args.attribute)
    print_trainable_parameters(model)
    print(model)
    count_parameter_bytes(model)
    #print("active3\n",model.active_adapters)


    # tokenizer.model_max_length = 2048
    # tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    # tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'

    print_trainable_parameters(model)
    print("total model parameters : {0}".format(count_model_parameters(model)))


    ##################
    # Data Processing
    ##################



    train_dataset = load_dataset_from_path_phi(args.train_dataset_path, args.attribute)
    test_dataset = load_dataset_from_path_phi(args.val_dataset_path, args.attribute)
    if args.test_dataset_size == -1:
        args.test_dataset_size = len(test_dataset)
    if args.train_dataset_size == -1:
        args.train_dataset_size = len(train_dataset)
    if args.val_dataset_size == -1:
        args.val_dataset_size = len(test_dataset)
    
    #get subset of the train dataset
    #train_dataset = train_dataset.select(range(args.train_dataset_size))
    #test_dataset = test_dataset.select(range(args.test_dataset_size))


    train_dataset = train_dataset.select(range(4))
    test_dataset = test_dataset.select(range(4))



    print("train dataset size", len(train_dataset))
    print("test dataset size", len(test_dataset))
    
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
    #processed_test_dataset = processed_test_dataset.filter(lambda x: len(tokenizer(x["messages_for_inference"], return_tensors="pt")["input_ids"]) <= max_sequence_length - max_new_tokens - 10)
    #print("after filtering the dataset size is : {0}".format(len(processed_test_dataset)))

    
    model.train()
    train_input_ids = torch.tensor(processed_train_dataset[0]['input_ids']).unsqueeze(0).to("cuda")
    labels = torch.tensor(processed_train_dataset[0]['labels']).unsqueeze(0).to("cuda")
    torch.set_float32_matmul_precision('high') # this is not the higest precision 32 : is seen as sum of 16 + 16
    print_trainable_parameters(model)
    scaler = torch.cuda.amp.GradScaler()
    print(model)
    for i in tqdm.tqdm(range(10)):
        with torch.autocast(device_type = "cuda", dtype = torch.float16):
            output = model(train_input_ids)
            loss = output.loss
            print(output)





    