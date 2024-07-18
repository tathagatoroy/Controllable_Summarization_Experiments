#using a config to file to test it out for now , will parameterize the code later
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, get_kbit_device_map
from utils import print_trainable_parameters, formatting_prompts_func, load_dataset_from_path, count_model_parameters
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
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, help="bnb 4 bit compute dtype", default = "torch.bfloat16")



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

    #overwrite the variables from config file
    model_name = args.model_name
    tokenizer_name = args.tokenizer_name
    train_dataset_path = args.train_dataset_path
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

    lora_config = LoraConfig(
        r = args.r, #rank
        lora_alpha = args.lora_alpha, #learning rate 
        target_modules=args.target_modules, #which matrices should use lora
        lora_dropout = args.lora_dropout, #dropout
        bias = args.bias, #bias
        task_type = args.task_type # task/model type
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size= args.per_device_train_batch_size, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        report_to=args.report_to,
        logging_steps=4,
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        fp16=True,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        disable_tqdm=False,
    )

    #set up wandb logging
    wandb.init(project="openelm_adapter", name=current_date, config=vars(args))
    wandb.config.update({"nf4_config": vars(nf4_config), "lora_config": vars(lora_config), "training_args": vars(training_args)})
    wandb.config.update({"model_name": model_name, "tokenizer_name": tokenizer_name, "train_dataset_path": train_dataset_path, "test_dataset_path": test_dataset_path, "val_dataset_path": val_dataset_path, "cache_dir": cache_dir, "instruction_template": instruction_template, "response_template": response_template})










    #load the dataset
    dataset = load_dataset_from_path(train_dataset_path, args.attribute)
    print("size of the dataset: ", len(dataset))

    #load accelerate 
    #accelerator = Accelerator()



    #load the model 
    print("loading the model and the tokenizer and preparing peft training")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    #load the model
    if args.load_previous_model:
        print("loading from " + args.previous_model_path)
        model = AutoModelForCausalLM.from_pretrained(args.previous_model_path, 
                                                trust_remote_code = True, 
                                                quantization_config = nf4_config, 
                                                torch_dtype = torch.float16,
                                                device_map = "cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(args.previous_model_path, trust_remote_code = True)
        config = PeftConfig.from_pretrained(args.previous_model_path)
        model = PeftModelForCausalLM.from_pretrained(model, args.previous_model_path, is_trainable = True)
        if args.use_current_adapter:
            adapter_name = list(model.peft_config.keys())[0]
            model.set_adapter(adapter_name)
            print("setting adapter: ", adapter_name)
    else:
        print("loading base model weights")
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                trust_remote_code = True, 
                                                quantization_config = nf4_config, 
                                                cache_dir = cache_dir,
                                                low_cpu_mem_usage = True,
                                                torch_dtype = torch.float16,
                                                device_map = "cuda:0")
    
        #load the tokenizer 
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code = True)
        tokenizer.padding_side = "right"
    model = prepare_model_for_kbit_training(model, nf4_config)
    model = get_peft_model(model, lora_config).cuda()


    #get the number of trainable paramaters 
    print_trainable_parameters(model)

    #define the data collator 
    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)


    #define the trainer
    print("defining the trainer")
    trainer = SFTTrainer(
        model=model,
        data_collator=collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        peft_config = lora_config,
        args = training_args,
        max_seq_length=4096,

    )

    #prepare the model optimizer for distributed training
    print("prepare the model for distributed training")
    #model, optimizer, train_loader, _ = accelerator.prepare(model, trainer.optimizer, trainer.get_train_dataloader(), None)
    model.train()

    




    #train the model
    print("training the model")
    trainer.train()
    
    
    










