import sys
import logging

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


"""
A simple example on using SFTTrainer and Accelerate to finetune Phi-3 models. For
a more advanced example, please follow HF alignment-handbook/scripts/run_sft.py.
This example has utilized DeepSpeed ZeRO3 offload to reduce the memory usage. The
script can be run on V100 or later generation GPUs. Here are some suggestions on 
futher reducing memory consumption:
    - reduce batch size
    - decrease lora dimension
    - restrict lora target modules
Please follow these steps to run the script:
1. Install dependencies: 
    conda install -c conda-forge accelerate
    pip3 install -i https://pypi.org/simple/ bitsandbytes
    pip3 install peft
    pip3 install deepspeed
2. Setup accelerate and deepspeed config based on the machine used:
    accelerate config
Here is a sample config for deepspeed zero3:
    compute_environment: LOCAL_MACHINE
    debug: false
    deepspeed_config:
    gradient_accumulation_steps: 1
    offload_optimizer_device: none
    offload_param_device: none
    zero3_init_flag: true
    zero3_save_16bit_model: true
    zero_stage: 3
    distributed_type: DEEPSPEED
    downcast_bf16: 'no'
    enable_cpu_affinity: false
    machine_rank: 0
    main_training_function: main
    mixed_precision: bf16
    num_machines: 1
    num_processes: 4
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false
3. check accelerate config:
    accelerate env
4. Run the code:
    accelerate launch sample_finetune.py
"""
if __name__ == "__main__":
        #parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="model name", default = "microsoft/Phi-3-mini-128k-instruct")
    parser.add_argument("--tokenizer_name", type=str, help="tokenizer name", default = "NousResearch/Llama-2-7b-hf")
    parser.add_argument("--train_dataset_path", type=str, help="train dataset path", default = "/home2/tathagato/summarization/MACSum/dataset/macdoc/train_dataset.json")
    parser.add_argument("--test_dataset_path", type=str, help="test dataset path", default = "/home2/tathagato/summarization/MACSum/dataset/macdoc/test_dataset.json")
    parser.add_argument("--val_dataset_path", type=str, help="val dataset path", default = "/home2/tathagato/summarization/MACSum/dataset/macdoc/val_dataset.json")
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
    parser.add_argument("--output_dir", type=str, help="output directory", default = "/scratch/tathagato/openelm_adapter experiments")
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

    args = parser.parse_args()
    #get the current date and time in human format
    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_" + str(args.attribute)
    #set the seed
    seed = args.seed
    torch.manual_seed(seed)
    args.output_dir = os.path.join(args.output_dir, current_date)
    logger = logging.getLogger(__name__)


    ###################
    # Hyper-parameters
    ###################
    training_config = {
        "bf16": False,
        "fp16" : True,
        "do_eval": False,
        "learning_rate": 5.0e-06,
        "log_level": "info",
        "logging_steps": 20,
        "logging_strategy": "steps",
        "lr_scheduler_type": "cosine",
        "num_train_epochs": 1,
        "max_steps": -1,
        "output_dir": "/scratch/tathagato/openelm_adapter_experiment",
        "overwrite_output_dir": True,
        "per_device_eval_batch_size": 1,
        "per_device_train_batch_size": 1,
        "remove_unused_columns": False,
        "save_steps": 100,
        "save_total_limit": 1,
        "seed": 0,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs":{"use_reentrant": False},
        "gradient_accumulation_steps": 1,
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
    checkpoint_path = "google/gemma-1.1-2b-it"
    # checkpoint_path = "microsoft/Phi-3-mini-128k-instruct"
    #use flash attention 1
    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=None,
        cache_dir = "/scratch/tathagato",
        attn_implementation = "eager",
        quantization_config = nf4_config, 

    )
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.model_max_length = 2048
    tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'

    print_trainable_parameters(model)
    print("total model parameters : {0}".format(count_model_parameters(model)))


    ##################
    # Data Processing
    ##################
    def apply_chat_template(
        example,
        tokenizer,
    ):
        messages = example["messages"]
        # Add an empty system message if there is none
        # if messages[0]["role"] != "system":
        #     messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        return example

    train_dataset = load_dataset_from_path_phi(args.train_dataset_path, args.attribute)
    test_dataset = load_dataset_from_path_phi(args.val_dataset_path, args.attribute)
    # raw_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", cache_dir = "/scratch/tathagato")
    # train_dataset = raw_dataset["train_sft"]
    # test_dataset = raw_dataset["test_sft"]
    # for key in train_dataset[0]:
    #     print(key)
    #     if key == "messages":
    #         for message in train_dataset[0][key]:
    #             print(message)
    #     else:
    #         print(train_dataset[0][key])
    #     print("\n\n")
    # column_names = list(train_dataset.features)
    # print(column_names)

    train_dataset = load_dataset_from_path_phi(args.train_dataset_path, 'length')
    column_names = list(train_dataset.features)

    print(len(train_dataset))

    processed_train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to train_sft",
    )

    processed_test_dataset = test_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to test_sft",
    )
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
        eval_dataset=processed_test_dataset,
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
    tokenizer.padding_side = 'left'
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(processed_test_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


    # ############
    # # Save model
    # ############
    trainer.save_model(train_conf.output_dir)