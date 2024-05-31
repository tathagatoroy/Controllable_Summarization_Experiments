import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from phi_dataset import create_huggingface_dataset_from_dictionary_for_phi , create_multiattribute_dataset_from_dictionary_for_phi
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import transformers
from utils import load_dataset_from_path_phi, load_multi_attribute_dataset_from_path,print_trainable_parameters, count_model_parameters
import argparse

from trl import SFTTrainer

if __name__ == "__main__":
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

    #set the qunatization config
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    #
    #Load the model and Tokenizer
    model_id = "google/gemma-2b-it"
    #
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0} , cache_dir = "/scratch/tathagato")
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)

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
    print(processed_train_dataset[0].keys())

    model = prepare_model_for_kbit_training(model)
    #
    target_modules = ['down_proj', 'k_proj', 'o_proj', 'gate_proj', 'q_proj', 'v_proj', 'up_proj']

    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )


    model = get_peft_model(model, lora_config)
    target_modules = ['down_proj', 'k_proj', 'o_proj', 'gate_proj', 'q_proj', 'v_proj', 'up_proj']
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side='right'
    torch.cuda.empty_cache()

    trainer = SFTTrainer(
        model=model,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_test_dataset,
        dataset_text_field="text",
        peft_config=lora_config,
        max_seq_length=2500,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=0.03,
            max_steps=100,
            learning_rate=2e-4,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
            save_strategy="epoch",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    #
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
