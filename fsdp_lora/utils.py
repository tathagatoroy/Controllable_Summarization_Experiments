import wandb 
from typing import Dict
import tqdm 
import torch 
import os
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from bitsandbytes.nn import Params4bit
from transformers.models.llama.modeling_llama import (
    LLAMA_ATTENTION_CLASSES,
    LlamaDecoderLayer,
    LlamaMLP,
)
from dora import BNBDORA, HQQDORA, DORALayer, MagnitudeLayer

from transformers.models.mistral.modeling_mistral import (
    MISTRAL_ATTENTION_CLASSES,
    MistralDecoderLayer,
    MistralMLP,
)
from torch import Tensor, nn
import torch.optim as optim
from torch.utils.data import DataLoader

import functools
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
import math

class Logger:
    def __init__(self, args, log_to="stdout", project_name="fsdp_qlora", entity=None, group=None, name=None, rank=0):
        # self.log_every_n_steps = log_every_n_steps TODO: add this back as an option
        self.log_to = log_to
        if self.log_to == "wandb" and rank==0:
            import wandb
            wandb.init(project=project_name, entity=entity, group=group, name=name, config=args)

    def log(self, d:Dict, rank:int):
        if rank != 0: return
        if self.log_to == "tqdm":
            for k,v in d.items():
                tqdm.write(f'{k}: {v}')
        elif self.log_to == "wandb":
            wandb.log(d)
        elif self.log_to == "stdout":
            for k,v in d.items():
                print(f'{k}: {v}')

    def finish(self, rank=0):
        if self.log_to == "wandb" and rank==0: wandb.finish()


def update_progress_bar(progress_bar:tqdm, epoch:int, log_loss:float, log_lr:float, rank:int):
    """Updates the progress bar with the current epoch, loss, and learning rate"""
    if rank == 0:
        if log_lr >=0:
            progress_bar.set_description(f"Epoch {epoch}, Loss {log_loss:.3f}, LR {log_lr:.2e}", refresh=True)
        else:
            progress_bar.set_description(f"Epoch {epoch}, Loss {log_loss:.3f}", refresh=True)

def n_loading_workers(quant_method:str, param_count:float):
    devprops = torch.cuda.get_device_properties(torch.cuda.current_device())
    left = int(os.cpu_count()/torch.cuda.device_count())
    right = int((4 if quant_method == "hqq" else 8) * (devprops.total_memory/1e9/40) * (70/(param_count/1e9)))
    return min(left, right)

def setup_quantized_meta_for_peft(model:nn.Module):
    """Replaces `quant_state.to` with a dummy function to prevent PEFT from moving `quant_state` to meta device"""
    def temp_to_method(self, *args, **kwargs):
        return self
    for param in model.parameters():
        if isinstance(param, Params4bit):
            param.quant_state._orig_to = param.quant_state.to
            param.quant_state.to = types.MethodType(temp_to_method, param.quant_state)

def setup_quantized_peft_meta_for_training(model:nn.Module):
    """Replaces dummy `quant_state.to` method with the original function to allow training to continue"""
    for param in model.parameters():
        if isinstance(param, Params4bit) and hasattr(param.quant_state, '_orig_to'):
            param.quant_state.to = param.quant_state._orig_to
            param.quant_state._orig_to = None

# LR scheduler.
def _get_cosine_one_cycle_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, min_lr_fraction = 0.1,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    scale_term = (1 - min_lr_fraction)
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return (math.cos(math.pi * progress)+1) * 0.5 * scale_term + min_lr_fraction

def get_cosine_one_cycle_scheduler(optimizer:optim.Optimizer, num_warmup_steps:int, num_training_steps:int, min_lr_fraction:float=0.1):
    "A more general cosine scheduler with to control the minimum learning rate"
    lr_lambda = functools.partial(
        _get_cosine_one_cycle_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_fraction=min_lr_fraction
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

def get_lr_scheduler(optimizer:optim.Optimizer, dataloader:DataLoader, gradient_accumulation_steps:int, args:Dict):
    """Returns linear, cosine, or constant learning rate scheduler"""
    num_training_steps = args['num_epochs'] * len(dataloader) // gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * 0.1)
    if args['lr_scheduler'] == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args['lr_scheduler'] == "cosine":
        lr_scheduler = get_cosine_one_cycle_scheduler(optimizer, num_warmup_steps, num_training_steps, min_lr_fraction=0.1)
    elif args['lr_scheduler'] == "constant":
        lr_scheduler = None
    else:
        raise NotImplementedError(f"{args['lr_scheduler']} LR scheduler not implemented yet")
    return lr_scheduler, num_training_steps


# Optimizer
def get_optimizer(model:nn.Module, args:Dict):
    """Returns an optimizer. We can add more options here if needed."""
    if args["optimizer"] in ["adam", "fused_adam"]:
        return optim.Adam(model.parameters(), lr=args['lr'], fused=args["optimizer"]=="fused_adam")
    elif args["optimizer"] == "sgd":
        return optim.SGD(model.parameters(), lr=args['lr'])
    elif args["optimizer"] == "adadelta":
        return optim.Adadelta(model.parameters(), lr=args['lr'])
    elif args["optimizer"] in ["adamw", "fused_adamw"]:
        return torch.optim.AdamW(model.parameters(), lr=args['lr'], betas=(0.9,0.95),
                                 eps=1e-5, weight_decay=args['wd'], fused=args["optimizer"]=="fused_adamw")
    else:
        raise ValueError("Invalid optimizer")

def set_requires_grad(module, value):
    for param in module.parameters():
        param.requires_grad = value
# Wrap the model using LoRA policy from llama-recipes or custom policy:
# This checks for lora layers (has weight and requires_grad)
def get_wrapping_policy(custom_policy:bool=False, vanilla_policy:bool=False):
    from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder

    if custom_policy:
        def lambda_policy_fn(module):
            # LoRA and DoRA trainable layers.
            return (isinstance(module, nn.Sequential) and all(m.weight.requires_grad for m in module)) or (isinstance(module, (DORALayer, MagnitudeLayer)))
    else:
        def lambda_policy_fn(module):
            return (
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )
    def self_attn_policy_fn(module):
        # Check module name is self_attn.
        return isinstance(module, tuple((*LLAMA_ATTENTION_CLASSES.values(), *MISTRAL_ATTENTION_CLASSES.values())))

    def mlp_policy_fn(module):
        # Check module name is self_attn.
        return isinstance(module, (LlamaMLP, MistralMLP))

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    self_attn_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=self_attn_policy_fn)
    mlp_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=mlp_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(LlamaDecoderLayer, MistralDecoderLayer),
    )
    if vanilla_policy:
        return transformer_wrap_policy

    policies=[lambda_policy, transformer_wrap_policy]
    if custom_policy:
        policies.extend([self_attn_policy, mlp_policy])
    return functools.partial(_or_policy, policies=policies)


def print_model_details(model, parent_name=''):
    """
    Recursively print details of a PyTorch model, including sub-modules and trainable parameters.

    Args:
        model (torch.nn.Module): The model to inspect.
        parent_name (str): The name of the parent module (used in recursion).
    """
    total_trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            total_trainable_params += param.numel()
        print(f"{name} | requires_grad: {param.requires_grad} | shape: {tuple(param.shape)} | dtype: {param.dtype}")
    print(f"{parent_name} | Total params: {total_params:,} | Trainable params: {total_trainable_params:,}")