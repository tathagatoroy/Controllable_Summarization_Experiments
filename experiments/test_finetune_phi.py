import sys
import logging
import tqdm
import datasets
from datasets import load_dataset
from peft import LoraConfig,PeftConfig, PeftModel, PeftModelForCausalLM
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
    parser.add_argument("--output_dir", type=str, help="output directory", default = "/scratch/tathagato/openelm_adapter_experiments")
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
    args = parser.parse_args()
    #get the current date and time in human format
    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_" + str(args.attribute)
    #set the seed
    seed = args.seed
    torch.manual_seed(seed)
    #args.output_dir = os.path.join(args.output_dir,args.attribute +  current_date)
    logger = logging.getLogger(__name__)


    ###################
    # Hyper-parameters
    ###################
    training_config = {
        "bf16": False,
        "fp16" : True,
        "do_eval": False,
        "learning_rate": 5e-4,
        "log_level": "info",
        "logging_steps": 20,
        "logging_strategy": "steps",
        "lr_scheduler_type": "cosine",
        "num_train_epochs": 5,
        "max_steps": -1,
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "per_device_eval_batch_size": 1,
        "per_device_train_batch_size": 1,
        "remove_unused_columns": False,
        "save_steps": 400,
        "save_total_limit": 400,
        "seed": 0,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs":{"use_reentrant": False},
        "gradient_accumulation_steps": 2,
        "warmup_ratio": 0.2,
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
    #print("active3\n",model.active_adapters)


    # tokenizer.model_max_length = 2048
    # tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    # tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'left'

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
    train_dataset = train_dataset.select(range(args.train_dataset_size))
    test_dataset = test_dataset.select(range(args.test_dataset_size))

    #train_dataset = train_dataset.select(range(12))


    print("train dataset size", len(train_dataset))
    print("test dataset size", len(test_dataset))


    column_names = []

    print(len(train_dataset))

    processed_train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to train_sft",
    )

    processed_test_dataset = test_dataset.map(
        apply_inference_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to test_sft",
    )
    print("tokenizer padding side", tokenizer.padding_side)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(processed_train_dataset[0]['text'])

    #running inference using pipeline
    # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, batch_size = 4)


    # # Initialize variables
    # prompts = []
    # batch_size = 4 # Adjust the batch size as needed
    # max_sequence_length = 2048  # Set this to your model's maximum sequence length
    # dist = {}
    # all_outputs = []
    # total_input_size = 0
    # if args.run_inference:
    #     inference_output = {}
    # for index, example in tqdm.tqdm(enumerate(processed_test_dataset)):
    #     messages = example["messages"]
    #     messages[-1]["content"] = ""
    #     prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
    #     # Check if the tokenized prompt length exceeds the maximum sequence length
    #     tokenized_prompt = pipe.tokenizer(prompt, return_tensors="pt")["input_ids"]
    #     prompt_length = tokenized_prompt.size(1)
    #     if prompt_length not in dist:
    #         dist[prompt_length] = 0
    #     dist[prompt_length] += 1
    #     #print(f"Example {index} has length {prompt_length}")
    #     if tokenized_prompt.size(1) > max_sequence_length:
    #         #print(f"Skipping example {index} due to length {tokenized_prompt.size(1)} > {max_sequence_length}")
    #         total_input_size += 1
    #         continue  # Skip this example if it exceeds the maximum sequence length

    #     prompts.append(prompt)

    #     # Check if we have reached the batch size
    #     if len(prompts) == batch_size:
    #         # Process the batch
    #         print("calling pipe")
    #         outputs = pipe(prompts, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            
    #         # Do something with the outputs
    #         print(len(outputs))  # Example: print the number of output
    #         print(outputs[0])

    #         # Reset the prompts list for the next batch
    #         prompts = []
    #         for output in outputs:
    #             all_outputs.append(output)
        

    # print("dist", dist)
    # print("total input size", total_input_size)
    # print("total outputs", len(all_outputs))
    # print("all outputs", all_outputs, file = open("sample_outputs","w"))
    # exit()




        







    #print("-----------------------------------------")
    #print(processed_train_dataset[0])
    #print("\n\n")
    # print(processed_test_dataset[0])
    ###########
    # Training
    ###########
    trainer = SFTTrainer(
        model=model,
        args=train_conf,
        peft_config=peft_conf,
        train_dataset=processed_train_dataset,
        max_seq_length=2048,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=True
    )
    print("is  model parallelism " ,trainer.args.parallel_mode)

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


    #############
    # Evaluation
    #############
    #print("tokenizer padding side", tokenizer.padding_side)
    #tokenizer.padding_side = 'left'
    # metrics = trainer.evaluate()
    # metrics["eval_samples"] = len(processed_test_dataset)
    # trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)


    # ############
    # # Save model
    # ############
    trainer.save_model(train_conf.output_dir)
    
    #merge and unload and save the model
    model = model.merge_and_unload()
    model.save_pretrained(os.path.join(train_conf.output_dir,"final_merged_model"))
