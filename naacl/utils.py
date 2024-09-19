#written with the help of chatgpt and claude 
import tqdm 
import torch
import transformers
from transformers import AutoModelForCausalLM
import time
from functools import wraps
import math 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.optim import AdamW
import os 
import wandb
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, PeftConfig, PeftModel


def load_peft_checkpoint(config, quantization_config, checkpoint_path):
    """ load a peft checkpoint """
    peft_config = PeftConfig.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(config['model_id'], quantization_config= quantization_config, use_cache = False, device_map = "cuda:0")
    model = PeftModel.from_pretrained(model = model, model_id = checkpoint_path, adapter_name= config['attributes'][0], is_trainable= False, config = peft_config)
    return model

def get_adapter_status(peft_model):
    print("model summary")
    model_status = peft_model.get_model_status()
    attributes = dir(model_status)
    for attr in attributes:
        if not attr.startswith("_"):
            print(f"{attr} : {getattr(model_status, attr)}")

    print("--------------------------------------------------------")


def print_layerwise_details(peft_model):
    for layer, param in peft_model.named_parameters():
        print(f"Layer: {layer} | Shape: {param.shape} | dtype: {param.dtype} | device: {param.device} | requires_grad: {param.requires_grad}")


def timer(func):
    # a decorator to measure the time taken by a function to execute
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds to complete.")
        return result
    return wrapper



def get_single_adapter_lora_model(base_model, lora_config , adapter_name):
    """ initialise a peft model with a single adapter with lora 
        Parameters:
        -----------
        base_model : quantized model
            The transformer model to be trained.
        lora_config : dict
            A dictionary containing the lora config
        adapter_name : str
               
    """
    model = get_peft_model(base_model, lora_config, adapter_name)
    #model = prepare_model_for_kbit_training(model) # not sure if this necessary , this is causing a bug for now disabling
    return model

def merge_configs(global_config, experiment_config):
    """ merge the global config and experiment config """
    updated_config = {}
    for key in global_config.keys():
        updated_config[key] = global_config[key]
    for key in experiment_config.keys():
        updated_config[key] = experiment_config[key]
    return updated_config

@timer
def generate_text(model, dataset, tokenizer = None, config=None):
    """
    Generate text summaries using a model for a given dataset.

    Parameters:
    -----------
    model : PreTrainedModel
        The model used for generating text.
    
    dataset : Iterable
        An iterable dataset where each item contains input data and references. pytorch dataset 
    
    tokenizer : PreTrainedTokenizer
        
    config : dict, optional
        Contains generation parameters like `do_sample`, `top_p`, `top_k`, `max_new_tokens`, and `num_return_sequences`.
    
    Returns:
    --------
    result_dict : dict
        A dictionary containing the generated text, input, predicted summary, reference, control attributes, and other relevant data.
    
    """
    
    result_dict = {}
    model.eval()
    with torch.no_grad():
        for index, item in tqdm.tqdm(enumerate(dataset), desc="Generating summaries"):
            # Move 'input_ids' to GPU
            new_item = {key: value.to('cuda') for key, value in item.items() if key == 'input_ids'}
            
            # Generate the output using the model
            output = model.generate(
                **new_item, 
                do_sample=config['do_sample'], 
                top_p=config['top_p'], 
                top_k=config['top_k'], 
                max_new_tokens=config['max_new_tokens'], 
                num_return_sequences=config['num_return_sequences']
            )
            
            # Decode the generated output
            decoded_text = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Separate the generation from the prompt
            generation_ids = output[0, len(new_item["input_ids"][0]):]
            generated_text = tokenizer.decode(generation_ids, skip_special_tokens=True)
            
            # Decode the prompt (input)
            prompt = tokenizer.decode(new_item["input_ids"][0], skip_special_tokens=True)
            
            # Store the results in the result_dict
            result_dict[index] = {
                'input': item['input'], 
                'predicted_summary': generated_text, 
                'reference': item['output'], 
                'generated_text': decoded_text, 
                'control_value': item['control_value'], 
                'control_attribute': item['control_attribute']
            }
            
            # Add any missing keys from the original item to the result
            for key in item.keys():
                if key not in result_dict[index]:
                    result_dict[index][key] = item[key]
            
            # Print the generated text for each item
            #print("generated_text:", generated_text)
    
    return result_dict

def collate_fn(batch):
    """ do huggingface style collation using torch dataset """
    return {key: [example[key] for example in batch] for key in batch[0].keys()}

import math

def get_lr(it, num_warmup_steps, num_training_steps, max_lr, min_lr):
    """
    Calculate the learning rate for the current training iteration based on a warmup phase 
    followed by a cosine decay.

    Parameters:
    -----------
    it : int
        The current training iteration (step).
    
    num_warmup_steps : int
        The number of warmup steps where the learning rate increases linearly from `min_lr` to `max_lr`.
        
    num_training_steps : int
        The total number of training steps, including the warmup phase.
    
    max_lr : float
        The maximum learning rate to be reached at the end of the warmup phase.
    
    min_lr : float
        The minimum learning rate used during the training and at the end of the cosine decay phase.
    
    Returns:
    --------
    float
        The calculated learning rate for the current iteration `it`.

    Example:
    --------
    >>> lr = get_lr(it=100, num_warmup_steps=500, num_training_steps=10000, max_lr=1e-3, min_lr=1e-6)
    """

    # Warmup phase: increase learning rate linearly from min_lr to max_lr over num_warmup_steps
    if it < num_warmup_steps:
        return min_lr + (max_lr - min_lr) * (it / num_warmup_steps)
    
    # Cosine decay phase: after the warmup, decay the learning rate following a cosine curve
    # Compute the progress of training after the warmup phase
    progress = (it - num_warmup_steps) / (num_training_steps - num_warmup_steps)
    
    # Apply cosine decay: learning rate decays from max_lr to min_lr following a cosine function
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def collate_function(tokenizer):
    """
    Returns a collate function for use in a DataLoader that dynamically pads input sequences 
    (input_ids, attention_mask, and labels) to the same length within a batch.

    This is typically used when working with variable-length sequences, such as tokenized text data, 
    where padding ensures that all sequences in a batch have the same length.

    Parameters:
    -----------
    tokenizer : PreTrainedTokenizer
        The tokenizer used for encoding input sequences. It is needed to identify the padding token 
        and to handle cases where the pad token might not be defined.

    Returns:
    --------
    collate_fn : function
        A function that can be passed to a PyTorch DataLoader for batch collation, padding the 
        input sequences to the same length.

    Example:
    --------
    >>> data_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_function(tokenizer))
    """

    def collate_fn(batch):
        """
        Pads input_ids, attention_mask, and labels to the same length for each batch.
        
        Parameters:
        -----------
        batch : list of dict
            A list of individual data points (in dictionary format) in a batch. Each dictionary should 
            contain 'input_ids', 'attention_mask', and 'labels'.
        
        Returns:
        --------
        dict
            A dictionary containing padded 'input_ids', 'attention_mask', and 'labels'.
        """

        # Extract input_ids, attention_mask, and labels from the batch
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]

        # Ensure that the tokenizer has a pad_token, otherwise use eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Pad the sequences to the same length for the batch
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        # Return the padded tensors as a dictionary
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels  # In this example, we assume labels are separate from input_ids
        }

    return collate_fn



@timer 
def train(model, tokenizer, train_dataset, eval_dataset, config=None, device=0, save_pretrained = True, do_wandb = False):
    """
    Trains a transformer model using gradient accumulation and cosine learning rate scheduling.

    Parameters:
    -----------
    model : PreTrainedModel
        The transformer model to be trained.

    tokenizer : PreTrainedTokenizer
        The tokenizer used for encoding inputs and decoding outputs.

    train_dataset : Dataset
        The dataset to train on. Each element should contain 'input_ids', 'attention_mask', and 'labels'.
    
    eval_dataset : Dataset
        The dataset to evaluate on. Each element should contain 'input_ids', 'attention_mask', and 'labels'.

    config : Namespace or dict
        A configuration object that contains training parameters such as:
        - `batch_size`
        - `gradient_accumulation_steps`
        - `learning_rate`
        - `max_grad_norm`
        - `warmup_ratio`
        - `max_lr`
        - `min_lr`
        - `logging_steps`
        - `eval_interval`
        - `num_epochs`
        - `output_dir`
    
    device : int, optional (default=0)
        The GPU device ID. If not available, falls back to CPU.
    
    Returns:
    --------
    None
        The function trains the model and periodically saves checkpoints.

    Example:
    --------
    >>> train(model, tokenizer, dataset, config, device=0)
    """

    # Set the device
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads() #https://github.com/huggingface/peft/issues/137#issuecomment-1445912413 otherwise it is breaking 
    # Set up the DataLoader for training
    dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=collate_function(tokenizer))
    
    # Set up optimizer (AdamW is commonly used for transformers)
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Prepare for training
    model.train()
    global_step = 0
    effective_batch_size = config['batch_size'] * config['gradient_accumulation_steps']
    total_examples = len(dataloader.dataset) * config['num_epochs']
    total_steps = (total_examples + config['batch_size'] - 1) // config['batch_size']
    effective_steps = (total_steps + config['gradient_accumulation_steps'] - 1) // config['gradient_accumulation_steps']
    warmup_steps = int(config['warmup_ratio'] * effective_steps)

    print(f"Starting training for attribute: {dataloader.dataset.attributes[0]}")
    print(f"Total steps: {total_steps} | Total Effective steps : {effective_steps} Warmup steps: {warmup_steps}")
    print(f"Effective batch size: {effective_batch_size} | Total examples: {total_examples}")


    total_loss = 0
    optimizer.zero_grad()

    # Manually create an iterator to allow resetting the dataloader
    dataloader_iter = iter(dataloader)
    effective_step_cnt = 0
    best_eval_loss = 1e12
    # Training loop
    with torch.autograd.detect_anomaly():
        for step in tqdm.tqdm(range(total_steps), total=total_steps, desc="Training"):
            # Reset the dataloader after going through it once
            if step % len(dataloader) == 0:
                dataloader_iter = iter(dataloader)  # Create a new iterator for the dataloader
            
            batch = next(dataloader_iter)  # Fetch the next batch from the iterator
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / config['gradient_accumulation_steps']
            total_loss += loss.item()
            
            # Backward pass and gradient accumulation

            loss.backward()
            
            if (step + 1) % config['gradient_accumulation_steps'] == 0 or step == total_steps - 1:
                effective_step_cnt += 1
                # Clip gradients to avoid exploding gradients
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                
                # Update learning rate using cosine decay
                lr = get_lr(effective_step_cnt, warmup_steps, effective_steps , config['max_lr'], config['min_lr'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # Step optimizer
                optimizer.step()
                optimizer.zero_grad()
                if do_wandb:
                    wandb.log({"loss": total_loss, "learning_rate": lr, "grad_norm": grad})
                
                print(f"Step: {effective_step_cnt} | Loss: {total_loss:.4f} | Learning Rate: {lr:.8f} | Grad Norm: {grad:.4f}")
                total_loss = 0
            
                # Save the model at every `logging_steps` interval
                if effective_steps % config['logging_steps'] == 0:
                    if save_pretrained:
                        model_save_path = os.path.join(config['output_dir'], f"model_{step}_{dataloader.dataset.attributes[0]}")
                        model.save_pretrained(model_save_path)
                        print(f"Model saved at {model_save_path}")

                    else:
                        model_save_path = os.path.join(config['output_dir'], f"model_{step}_{dataloader.dataset.attributes[0]}.pt")
                        torch.save(model.state_dict(), model_save_path)
                        print(f"Model saved at {model_save_path}")
                
                # Evaluate the model at every `eval_interval`
                if effective_steps % config['eval_interval'] == 0:
                    eval_loss = evaluate(model, tokenizer, eval_dataset, config, device)
                    if do_wandb:
                        wandb.log({"eval_loss": eval_loss})
                    print(f"Eval Loss at step {step}: {eval_loss:.4f}")
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        if save_pretrained:
                            model_save_path = os.path.join(config['output_dir'], f"best_model_{dataloader.dataset.attributes[0]}")
                            model.save_pretrained(model_save_path)
                            print(f"Model saved at {model_save_path}")
                        else:
                            model_save_path = os.path.join(config['output_dir'], f"best_model_{dataloader.dataset.attributes[0]}.pt")
                            torch.save(model.state_dict(), model_save_path)
                            print(f"Model saved at {model_save_path}")
                        print(f"Best Model saved at {model_save_path}")
                        print(f"Best Eval Loss: {best_eval_loss:.4f}")
                        print(f"Best Step: {effective_step_cnt}")
                
    print("training done")
    if save_pretrained:
        model_save_path = os.path.join(config['output_dir'], f"final_model_{dataloader.dataset.attributes[0]}")
        model.save_pretrained(model_save_path, safe_serialization=False)
    else:
        model_save_path = os.path.join(config['output_dir'], f"final_model_{dataloader.dataset.attributes[0]}.pt")
        torch.save(model.state_dict(), model_save_path)
    print(f"Final Model saved at {model_save_path}")
    


@timer
def evaluate(model, tokenizer, dataset, config = None, device=0):
    """
    Evaluates the model by computing the evaluation loss on the provided dataset.

    Parameters:
    -----------
    model : PreTrainedModel
        The transformer model to be evaluated.

    tokenizer : PreTrainedTokenizer
        The tokenizer used for encoding inputs.

    dataset : Dataset
        The dataset to evaluate on.

    config : Namespace or dict
        A configuration object containing evaluation parameters such as batch_size.

    device : int, optional (default=0)
        The GPU device ID. If not available, falls back to CPU.

    Returns:
    --------
    float
        The average evaluation loss.

    Example:
    --------
    >>> eval_loss = evaluate(model, tokenizer, dataset, config, device=0)
    """

    # Set the model to evaluation mode
    model.eval()
    eval_dataloader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=collate_function(tokenizer))
    
    total_eval_loss = 0
    total_steps = len(eval_dataloader)
    
    with torch.no_grad():
        for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_eval_loss += loss.item()
    
    avg_eval_loss = total_eval_loss / total_steps
    model.train()  # Return to training mode after evaluation

    return avg_eval_loss
