import torch 
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from transformers import TrainingArguments
from trl import ModelConfig




#model config
model_name =  "apple/OpenELM-450M-Instruct"

#tokenizer config
tokenizer_name = "NousResearch/Llama-2-7b-hf"
    

#dataset config
val_dataset_path = "/home2/tathagato/summarization/MACSum/dataset/macdoc/val_dataset.json"
train_dataset_path = "/home2/tathagato/summarization/MACSum/dataset/macdoc/train_dataset.json"
test_dataset_path = "/home2/tathagato/summarization/MACSum/dataset/macdoc/test_dataset.json"


#bits and bytes config
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

#lora config 
lora_config = LoraConfig(
    r = 32, #rank
    lora_alpha = 16, #learning rate 
    target_modules=["qkv_proj"], #which matrices should use lora
    lora_dropout = 0.1, #dropout
    bias = "none", #bias
    task_type = "CAUSAL_LM" # task/model type
)

#system config 
cache_dir = "/scratch/tathagato"


#prompt config 
response_template = "### Response:"
instruction_template = "### Instruction:"


#training config 
training_args = TrainingArguments(
    output_dir="/scratch/tathagato/openelm_single_attribute_adapter",
    num_train_epochs=10,
    per_device_train_batch_size= 1, 
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    # optim="paged_adamw_32bit",
    
    #torch_compile=True, # optimizations
    # optim="adamw_torch_fused", # improved optimizer 
    optim="adamw_bnb_8bit", #     #['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla', 'adamw_apex_fused', 'adafactor', 'adamw_bnb_8bit', 'adamw_anyprecision', 'sgd', 'adagrad']
    report_to="wandb",
    
    logging_steps=4,
    save_strategy="epoch",
    learning_rate=1e-5,
    #bf16=True,
    fp16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm= False, #True # disable tqdm since with packing values are in correct
)


