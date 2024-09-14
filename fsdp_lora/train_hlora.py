import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from bitsandbytes.nn import Linear4bit
import sys
sys.path.append("./scripts")
from hlora import HLORAConfig, replace_linear4bit_with_hlora, set_train_adapters, set_inference_adapters, set_gradients_on_the_model, print_model_layer_info, replace_lora_with_hlora
from dataset import MACSUM
from torch.nn.utils.rnn import pad_sequence
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import json 
import argparse
import tqdm
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TOKENIZERS_PARALLELISM'] = "false"
import pickle as pkl
import copy
from transformers import pipeline
import time


#running command using 
# export CUDA_VISIBLE_DEVICES=0, python train_hlora.py --debug > output.txt
# CUDA_VISIBLE_DEVICES=0 python train_hlora.py --output_dir /scratch/tathagato/length_then_extractiveness_llama_test --attribute_1 length --attribute_2 extractiveness --model_id akjindal53244/Llama-3.1-Storm-8B --debug > test.txt 2>&1


def collate_fn(batch):
    return {key: [example[key] for example in batch] for key in batch[0].keys()}
def generate_text_with_pipeline(model, dataset, batch_size=8, max_length=2048, args=None):

    result_dict = {}
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for index, item in tqdm.tqdm(enumerate(dataset), desc="Generating summaries"):
            new_item = {key: value.to('cuda') for key, value in item.items() if key == 'input_ids'}
            output = model.generate(**new_item, do_sample=args.do_sample, top_p=args.top_p, top_k=args.top_k, max_new_tokens=args.max_new_tokens, num_return_sequences=args.num_return_sequences)
            
            decoded_text = tokenizer.decode(output[0], skip_special_tokens=True)
            generation_ids = output[0,len(new_item["input_ids"][0]):]
            generated_text = tokenizer.decode(generation_ids, skip_special_tokens=True)
            prompt = tokenizer.decode(new_item["input_ids"][0], skip_special_tokens=True)
            #print(item.keys())
            result_dict[index] = {'input': item['input'], 'predicted_summary': generated_text, 'reference': item['output'], 'generated_text' : decoded_text, 'control_value' : item['control_value'], 'control_attribute' : item['control_attribute']}
            for key in item.keys():
                if key not in result_dict[index].keys():
                    result_dict[index][key] = item[key]
            print("generated_text: ", generated_text)  
    return result_dict


    # #for debug 
    
    # # Create a DataLoader for batching
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # all_generated_texts = []
    
    # for batch in tqdm.tqdm(dataloader):
    #     # Extract the prompts from the batch
    #     prompts = batch['prompt']
    #     for i in range(len(prompts)):
    #         print(prompts[i])
        
    #     # Generate text for the batch
    #     outputs = generator(prompts, do_sample=args.do_sample, top_p=args.top_p, top_k=args.top_k, max_new_tokens=args.max_new_tokens, num_return_sequences=1)
        
    #     # Extract the generated text from the outputs
    #     generated_texts = [output[0]['generated_text'] for output in outputs]
    #     #predicted_summaries = [text.split("\n\n")[-1] for text in generated_texts]
        
    #     # Store the results
    #     for i, generated_text in enumerate(generated_texts):
    #         all_generated_texts.append({
    #             'input': batch['input'][i], #src text 
    #             'prompt': batch['prompt'][i], #prompt used 
    #             'generated_text': generated_text, # generated text
    #             'reference': batch['output'][i], #reference text
    #             'control_value': batch['control_value'][i], #control value
    #             'control_attribute': batch['control_attribute'][i] #control attribute
    #             #'predicted_summary' : predicted_summaries[i] #predicted summary 
    #             })
    
    # return all_generated_texts
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

def print_trainable_parameters(model):
    print(f"Trainable parameters: {count_trainable_parameters(model)} | Total parameters: {count_total_parameters(model)} | Trainable parameter ratio: {count_trainable_parameters(model) / count_total_parameters(model)}")

def get_detached_state_dict(model):
    # Get the state dict of the model
    state_dict = model.state_dict()
    
    # Create a deep copy of the state dict
    detached_state_dict = copy.deepcopy(state_dict)
    
    # Detach all tensors in the copied state dict
    for key, value in detached_state_dict.items():
        if isinstance(value, torch.Tensor):
            detached_state_dict[key] = value.detach().clone().cpu()
    
    return detached_state_dict

def print_layers_with_requires_grad(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} is trainable")
        else:
            print(f"{name} is not trainable")

def return_changed_layers(old_state_dict, new_state_dict):
    # ensure that the keys are the same
    assert old_state_dict.keys() == new_state_dict.keys()
    changed_layers = {}
    for key in old_state_dict.keys():
        old_state_dict_params = old_state_dict[key]
        new_state_dict_params = new_state_dict[key]
        if torch.allclose(old_state_dict_params, new_state_dict_params):
            continue
        else:
            changed_layers[key] = (old_state_dict_params, new_state_dict_params)
            print(f"Layer {key} has changed")
    return changed_layers


def get_lr(it, num_warmup_steps, num_training_steps, max_lr, min_lr):
    # Warmup phase
    if it < num_warmup_steps:
        return min_lr + (max_lr - min_lr) * (it / num_warmup_steps)
    
    # Cosine decay phase
    progress = (it - num_warmup_steps) / (num_training_steps - num_warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def setup_hlora_model(model_id: str, lora_config: LoraConfig, hlora_config: HLORAConfig, bnb_config: BitsAndBytesConfig):
    # 1. Load the model
    model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config = bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 2. Get PEFT model
    peft_model = get_peft_model(model, lora_config)


    # 3. Replace lora layers with HLORA
    peft_model = replace_lora_with_hlora(peft_model, hlora_config)



    # 4. Set which adapters to use for training, inference and gradients    
    # hlora_model= set_train_adapters(hlora_model, level_1=True, level_2=False)
    # hlora_model = set_inference_adapters(hlora_model, level_1=True, level_2=False)
    # hlora_model = set_gradients_on_the_model(hlora_model)

    #print_model_layer_info(hlora_model)
    prepare_model_for_kbit_training(peft_model)

    return peft_model, tokenizer

def collate_function(tokenizer):
    def collate_fn(batch):
        # Example of how to use the tokenizer within the collate function
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Pad sequences
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-1)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids  # Assuming labels are the same as input_ids for this example
        }
    return collate_fn
def train(model, tokenizer, dataset, args = None , device = 0):

    
    # Set the device
    torch.cuda.set_device(device)
    device = torch.device("cuda", device)
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Set up the dataset and dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_function(tokenizer))
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    
    # Training loop
    model.train()
    global_step = 0
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    total_examples = len(dataloader.dataset) * args.num_epochs
    all_steps = total_examples // args.batch_size if total_examples % args.batch_size == 0 else total_examples // args.batch_size + 1
    total_steps = total_examples // effective_batch_size if total_examples % effective_batch_size == 0 else total_examples // effective_batch_size + 1
    warmup_steps = args.warmup_ratio * total_steps

    print("starting training for attribute ", dataloader.dataset.attribute)
    print(f"Total effective steps: {total_steps} | Warmup steps: {warmup_steps} | total forward steps {all_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"total examples: {total_examples}")

    total_loss = 0
    optimizer.zero_grad()
    for idx, step in tqdm.tqdm(enumerate(range(all_steps)), total=all_steps):
        batch = next(iter(dataloader))
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / args.gradient_accumulation_steps
        total_loss += loss.item()
        
        loss.backward()
            
        if (step + 1) % args.gradient_accumulation_steps == 0 or step == total_steps - 1:
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            lr = get_lr(step, warmup_steps, total_steps, args.max_lr, args.min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
        
            print(f" Step: {step} | Loss: {total_loss} | learning rate: {lr} | Grad norm: {grad}")
            optimizer.zero_grad()

            total_loss = 0
        # Save model at the end of the every x steps
        if (step + 1) % args.logging_steps == 0:
            model_save_path = os.path.join(args.output_dir, f"model_{step}_{dataloader.dataset.attribute}.pt")
            #model.save_pretrained(model_save_path)   
            #dump the state dict
            torch.save(model.state_dict(), model_save_path) 
            print(f"model saved at {model_save_path}")



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HLORA Training Script")
    
    # Model and data arguments
    parser.add_argument("--model_id", type=str, default="akjindal53244/Llama-3.1-Storm-8B")
    parser.add_argument("--train_dataset_path", type=str, default="/home2/tathagato/summarization/MACSUM/dataset/macdoc/train_dataset.json")
    parser.add_argument("--test_dataset_path", type=str, default="/home2/tathagato/summarization/MACSUM/dataset/macdoc/test_dataset.json")

    parser.add_argument("--output_dir", type=str, default="/scratch/tathagato/hlora_train_length_then_extractiveness", help="Directory to save model checkpoints")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of steps for gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("--max_lr", type=float, default=5e-5, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of total steps for warmup")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--logging_steps", type=int, default=300, help="Number of steps between model saves")
    
    # LORA and HLORA configuration arguments
    parser.add_argument("--lora_r", type=int, default=64, help="Rank of the LORA adapter")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha parameter for LORA")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability for LORA")
    parser.add_argument("--hlora_rank_1", type=int, default=64, help="Rank of HLORA rank 1")
    parser.add_argument("--hlora_rank_2", type=int, default=32, help="Rank of HLORA rank 2")
    parser.add_argument("--hlora_alpha_1", type=int, default=32, help="Alpha parameter for HLORA rank 1")
    parser.add_argument("--hlora_alpha_2", type=int, default=16, help="Alpha parameter for HLORA rank 2")

    #attributes 
    parser.add_argument("--attribute_1", type=str, default="length", help="attribute 1")
    parser.add_argument("--attribute_2", type=str, default="extractiveness", help="attribute 2")

    #generation arguments
    parser.add_argument('--do_sample', type=bool, default=True, help="Whether to sample during text generation.")
    parser.add_argument('--top_p', type=float, default=0.95, help="Top-p for nucleus sampling.")
    parser.add_argument('--top_k', type=int, default=50, help="Top-k for top-k sampling.")
    parser.add_argument('--max_new_tokens', type=int, default=500, help="Maximum number of new tokens to generate.")
    parser.add_argument('--num_return_sequences', type=int, default=1, help="Number of sequences to return during generation.")


    #debug 
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    #Initialize LORA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 4. Define HLORA config
    hlora_config = HLORAConfig(
        lora_rank_1=args.hlora_rank_1,
        lora_rank_2=args.hlora_rank_2,
        lora_alpha_1=args.hlora_alpha_1,
        lora_alpha_2=args.hlora_alpha_2,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )    
    #base_model = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=bnb_config, device_map="cuda") #use to initialise the pipeline 

    #load the model 
    model, tokenizer = setup_hlora_model(args.model_id, lora_config, hlora_config, bnb_config)
    model = set_train_adapters(model, level_1=True, level_2=False)
    model = set_inference_adapters(model, level_1=True, level_2=False)
    model = set_gradients_on_the_model(model)

    print_trainable_parameters(model)

    #print_layers_with_requires_grad(model)
    model_type = "llama31"
    if "mistral" in args.model_id:
        model_type = "mistral"


    #load the two dataset 
    if args.debug:
        first_dataset = MACSUM(args.train_dataset_path, attribute = args.attribute_1, tokenizer = tokenizer, mode = 'train', model_type= model_type, size = 32)
        second_dataset = MACSUM(args.train_dataset_path, attribute = args.attribute_2, tokenizer = tokenizer, mode = 'train', model_type= model_type, size = 32)
        val_attribute_1_dataset = MACSUM(args.test_dataset_path, attribute = args.attribute_1, tokenizer = tokenizer, mode = 'inference', model_type= model_type, size = 2)
        val_attribute_2_dataset = MACSUM(args.test_dataset_path, attribute = args.attribute_2, tokenizer = tokenizer, mode = 'inference', model_type= model_type, size = 2)
    else:
        first_dataset = MACSUM(args.train_dataset_path, attribute = args.attribute_1, tokenizer = tokenizer, mode = 'train', model_type= model_type)
        second_dataset = MACSUM(args.train_dataset_path, attribute = args.attribute_2, tokenizer = tokenizer, mode = 'train', model_type= model_type)
        val_attribute_1_dataset = MACSUM(args.test_dataset_path, attribute = args.attribute_1, tokenizer = tokenizer, mode = 'inference', model_type= model_type)
        val_attribute_2_dataset = MACSUM(args.test_dataset_path, attribute = args.attribute_2, tokenizer = tokenizer, mode = 'inference', model_type= model_type)

    
    print("First dataset length: ", len(first_dataset))
    print("Second dataset length: ", len(second_dataset))
    print("Val attribute 1 dataset length: ", len(val_attribute_1_dataset))
    print("Val attribute 2 dataset length: ", len(val_attribute_2_dataset))
    #print_model_layer_info(model)
    #print_layers_with_requires_grad(model)
    initial_state_dict = get_detached_state_dict(model)
    
    # start_time = time.time()
    # generator = pipeline('text-generation', model=base_model, tokenizer = tokenizer)
    # all_generated_texts_attribute_1 = generate_text_with_pipeline(generator, val_attribute_1_dataset, args=args)
    # all_generated_texts_attribute_2 = generate_text_with_pipeline(generator, val_attribute_2_dataset, args=args)
    # save_path_attribute_1 = os.path.join(args.output_dir, f"zero_shot_generated_texts_{args.attribute_1}.pkl")
    # save_path_attribute_2 = os.path.join(args.output_dir, f"zero_shot_generated_texts_{args.attribute_2}.pkl")
    # with open(save_path_attribute_1, "wb") as f:
    #     pkl.dump(all_generated_texts_attribute_1, f)
    #     print(f"Zero shot generated texts for {args.attribute_1} saved at {save_path_attribute_1}")
    # with open(save_path_attribute_2, "wb") as f:
    #     pkl.dump(all_generated_texts_attribute_2, f)
    #     print(f"Zero shot generated texts for {args.attribute_2} saved at {save_path_attribute_2}")
    # end_time = time.time()
    # print(f"Zero shot generation took {end_time - start_time} seconds for {len(val_attribute_1_dataset) + len(val_attribute_2_dataset)} examples")

    #import pdb; pdb.set_trace()
    #train the model for the first attribute
    start_time = time.time()
    train(model, tokenizer, first_dataset, args = args, device = 0)
    end_time = time.time()
    print(f"Training for {args.attribute_1} took {end_time - start_time} seconds for {len(first_dataset)} examples")
    intermediate_state_dict = get_detached_state_dict(model)

    start_time = time.time()
    # generator = pipeline('text-generation', model=base_model, tokenizer = tokenizer)
    # generator.model = model
    all_generated_texts_attribute_1 = generate_text_with_pipeline(model, val_attribute_1_dataset, args=args)
    all_generated_texts_attribute_2 = generate_text_with_pipeline(model, val_attribute_2_dataset, args=args)
    save_path_attribute_1 = os.path.join(args.output_dir, f"attribute_1_{args.attribute_1}_finetune_generated_texts_{args.attribute_1}.pkl")
    save_path_attribute_2 = os.path.join(args.output_dir, f"attribute_1_{args.attribute_1}_finetune_generated_texts_{args.attribute_2}.pkl")
    with open(save_path_attribute_1, "wb") as f:
        pkl.dump(all_generated_texts_attribute_1, f)
        print(f"{args.attribute_1} generated texts for {args.attribute_1} saved at {save_path_attribute_1}")
    with open(save_path_attribute_2, "wb") as f:
        pkl.dump(all_generated_texts_attribute_2, f)
        print(f"{args.attribute_1} generated texts for {args.attribute_2} saved at {save_path_attribute_2}")
    end_time = time.time()
    print(f"Finetuning generation took {end_time - start_time} seconds for {len(val_attribute_1_dataset) + len(val_attribute_2_dataset)} examples")

    #save the intermediate model 
    model_save_path = os.path.join(args.output_dir, f"model_intermediate_{args.attribute_1}.pt")
    #model.save_pretrained(model_save_path)
    torch.save(model.state_dict(), model_save_path)
    print(f"model saved at {model_save_path}")

    #set the second adapter to be trainable
    model = set_train_adapters(model, level_1=False, level_2=True)
    model = set_inference_adapters(model, level_1=True, level_2=True)
    model = set_gradients_on_the_model(model)

    print_layers_with_requires_grad(model)
    print_trainable_parameters(model)


    #print_model_layer_info(model)
    start_time = time.time()
    train(model, tokenizer, second_dataset, args = args, device = 0)
    end_time = time.time()
    print(f"Training for {args.attribute_2} took {end_time - start_time} seconds for {len(second_dataset)} examples")
    final_state_dict = get_detached_state_dict(model)

    start_time = time.time()
    # generator = pipeline('text-generation', model=base_model, tokenizer = tokenizer)
    # generator.model = model
    all_generated_texts_attribute_1 = generate_text_with_pipeline(model, val_attribute_1_dataset, args=args)
    all_generated_texts_attribute_2 = generate_text_with_pipeline(model, val_attribute_2_dataset, args=args)
    save_path_attribute_1 = os.path.join(args.output_dir, f"attribute_1_{args.attribute_1}_and_attribute_2_{args.attribute_2}_finetune_generated_texts_{args.attribute_1}.pkl")
    save_path_attribute_2 = os.path.join(args.output_dir, f"attribute_1_{args.attribute_1}_and_attribute_2_{args.attribute_2}_finetune_generated_texts_{args.attribute_2}.pkl")
    with open(save_path_attribute_1, "wb") as f:
        pkl.dump(all_generated_texts_attribute_1, f)
        print(f"{args.attribute_1} generated texts for {args.attribute_1} saved at {save_path_attribute_1}")
    with open(save_path_attribute_2, "wb") as f:
        pkl.dump(all_generated_texts_attribute_2, f)
        print(f"{args.attribute_1} generated texts for {args.attribute_2} saved at {save_path_attribute_2}")
    end_time = time.time()
    print(f"Finetuning generation took {end_time - start_time} seconds for {len(val_attribute_1_dataset) + len(val_attribute_2_dataset)} examples")

    print("Checking the changed layers between the initial and intermediate state dict")
    changed_layers = return_changed_layers(initial_state_dict, intermediate_state_dict)

    print("---------------------------------------------------------------------------")
    print("Checking the changed layers between the intermediate and final state dict")
    changed_layers = return_changed_layers(intermediate_state_dict, final_state_dict)
    

    #save the final model 
    model_save_path = os.path.join(args.output_dir, f"model_final_{args.attribute_1}_{args.attribute_2}.pt")
    #model.save_pretrained(model_save_path)
    torch.save(model.state_dict(), model_save_path) 
    print(f"model saved at {model_save_path}")

    #save lora and hlora config and args as final_config 
    final_config = {
        "lora_config": lora_config,
        "hlora_config": hlora_config,
        "args": args
    }
    #dump it as a json file
    with open(os.path.join(args.output_dir, "final_config.pkl"), 'wb') as f:
        pkl.dump(final_config, f)




