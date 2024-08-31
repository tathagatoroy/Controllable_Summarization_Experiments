"""
Read our announcement blog post: https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html.

This script trains a model using FSDP with LoRA & QLoRA. It pulls inspiration from
- llama-recipes (https://github.com/facebookresearch/llama-recipes/blob/main/src/llama_recipes/finetuning.py)
- PyTorch FSDP docs (https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- bitsandbytes (https://github.com/TimDettmers/bitsandbytes)

For information on the different arguments, run `python train.py --help`

You should treat this script as an alpha/preview release. If you're not comfortable with testing and debugging
models, we'd suggest holding off for a few months while the community more fully tests the approach.
"""

# Imports

# General
import copy
import functools
import gc
import math
import os
import sys
import time
import types
from contextlib import nullcontext
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import bitsandbytes as bnb
import safetensors
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from accelerate import init_empty_weights
from accelerate.utils import set_seed

# Model loading
from bitsandbytes.nn import Linear4bit, Params4bit
from fastcore.parallel import parallel

# Argument parsing
from fastcore.script import Param, bool_arg, call_parse
from packaging.version import parse
from safetensors.torch import save_file

# Torch + distributed training
from torch import Tensor, nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
    offload_wrapper,
)

# FSDP
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import BackwardPrefetch, CPUOffload, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, hub
from dataset import MACSUM


# To add a new model, import the transformer, attention, & MLP layers
# for the wrapping policy and `check_fn` in activation checkpointing
from transformers.models.llama.modeling_llama import (
    LLAMA_ATTENTION_CLASSES,
    LlamaDecoderLayer,
    LlamaMLP,
)
from transformers.models.mistral.modeling_mistral import (
    MISTRAL_ATTENTION_CLASSES,
    MistralDecoderLayer,
    MistralMLP,
)

# To get rid of tokenizers warnings for now
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# For logging things during training
try:
    import wandb
except ImportError:
    pass

# LoRA and DORA modules
sys.path.append("./scripts")
from lora import LORA
from hlora import HLORAPeftModel,HLORAConfig, HLORA, replace_linear4bit_with_hlora
from utils import *
from profiling_utils import profiling_context



def train(model, dataloader, args, logger, rank = 0):
    # Create the optimizer
    optimizer = get_optimizer(model, args)

    # LR scheduler.
    gradient_accumulation_steps = max(1, args['gradient_accumulation_steps'])
    lr_scheduler, num_training_steps = get_lr_scheduler(optimizer, dataloader, gradient_accumulation_steps, args)

    # Sanity check: see what parameters the optimizer has and which require grad:
    if rank == 0 and args['verbose']:
        print("Optimizer params:")
        for group in optimizer.param_groups:
            for param in group['params']:
                print(f"Shape: {param.shape}, Requires Grad: {param.requires_grad}, Dtype: {param.dtype}")


    # Autocast for mixed precision with fp16/bf16 compute types with fp32 params
    if args["precision"] in ["fp16_autocast", "bf16_autocast", "bf16_buffers_autocast"]:
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=compute_dtype)
    else:
        autocast = nullcontext()
    scaler = ShardedGradScaler() if args["precision"] == "fp16_autocast" else None
    scale_grads = scaler is not None


    if rank == 0:
        print("Total Training Steps:", num_training_steps)
    memory_stats = []
    progress_bar = tqdm(range(num_training_steps), disable=rank != 0)
    init_start_event.record()
    log_loss, log_lr = 0.0, -1
    # Reset peak memory to track that
    torch.cuda.reset_peak_memory_stats(local_rank)
    with torch.autograd.detect_anomaly():

        with profiling_context(args, rank=rank) as prof:
            for epoch in range(args['num_epochs']):
                update_progress_bar(progress_bar, epoch, log_loss, log_lr, rank)
                model.train()
                ddp_loss = torch.zeros(2).to(local_rank)

                for batch_idx, batch in enumerate(dataloader):

                    accumulate_grads = (batch_idx+1) % gradient_accumulation_steps == 0

                    # Prevent gradient syncing until update step if using no_sync option.
                    # Documentation states this should only be used on the root FSDP instance
                    # We assume this is a one-node setup
                    if args['no_sync'] and not accumulate_grads:
                        sync_context = model.no_sync()
                    else:
                        sync_context = nullcontext()

                    # Start logging memory (first iter) if requested
                    if args['profile_memory'] and batch_idx==0 and rank == 0 and epoch == 0:
                        torch.cuda.memory._record_memory_history()

                    # Log memory usage
                    if batch_idx == 0 and epoch == 0 and (rank == 0 or args['verbose']):
                        reserved_before_forward = torch.cuda.memory_reserved(local_rank)
                        memory_stats.append(f"Rank {rank}: Before forward: {reserved_before_forward/2**30:.2f} GiB")
                        if args["log_to"] == 'wandb':
                            logger.log({"memory/allocated_before_forward": torch.cuda.memory_allocated(local_rank)}, rank)
                            logger.log({"memory/reserved_before_forward": reserved_before_forward}, rank)

                    # Forward pass
                    with sync_context:
                        with autocast:
                            output = model(
                                batch['input_ids'].to(local_rank),
                                labels=batch['labels'].to(local_rank),
                                attention_mask=None,
                            )
                            loss = output.loss

                        # Scale loss for gradient accumulation
                        loss = loss / gradient_accumulation_steps

                        # Log memory usage
                        if batch_idx == 0 and epoch == 0 and (rank == 0 or args['verbose']):
                            reserved_after_forward = torch.cuda.memory_reserved(local_rank)
                            memory_stats.append(f"Rank {rank}: After forward: {reserved_after_forward/2**30:.2f} GiB")
                            if args["log_to"] == 'wandb':
                                logger.log({"memory/allocated_after_forward": torch.cuda.memory_allocated(local_rank)}, rank)
                                logger.log({"memory/reserved_after_forward": reserved_after_forward}, rank)

                        # Backward pass
                        if scale_grads:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                    # Record loss
                    bs = batch['input_ids'].shape[0]
                    ddp_loss[0] += loss.item() * bs * gradient_accumulation_steps
                    ddp_loss[1] += bs

                    # Step the optimizer (w/ gradient accumulation)
                    if accumulate_grads:
                        if args['apply_gradient_clipping'] and (args['grad_norm'] is not None):
                            current_norm = model.clip_grad_norm_(args['grad_norm'], norm_type=2.0)
                            if rank == 0 and args['verbose']:
                                print(f"Gradient norm: {current_norm}")
                            if args["log_to"] == 'wandb':
                                logger.log({"grad_norm": current_norm}, rank)
                        if scale_grads:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()
                        # avoid overhead when lr is constant.
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                        progress_bar.update(1)

                    # Log memory usage after backward
                    if batch_idx == 0 and epoch == 0 and (rank == 0 or args['verbose']):
                        reserved_after_backward = torch.cuda.memory_reserved(local_rank)
                        memory_stats.append(f"Rank {rank}: After backward: {reserved_after_backward/2**30:.2f} GiB")
                        if args["log_to"] == 'wandb':
                            logger.log({"memory/allocated_after_backward": torch.cuda.memory_allocated(local_rank)}, rank)
                            logger.log({"memory/reserved_after_backward": reserved_after_backward}, rank)

                    # Delete the output so more memory frees up before the next forward pass
                    output = None
                    loss = None

                    # Stop logging memory (first iter)
                    if args['profile_memory'] and batch_idx==0 and rank == 0 and epoch == 0:
                        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
                        torch.cuda.memory._record_memory_history(enabled=None) # Stop recording

                    # Log loss every gradient update steps
                    if accumulate_grads:
                        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
                        if rank == 0:
                            log_loss = ddp_loss[0] / ddp_loss[1]
                            if lr_scheduler is not None:
                                log_lr = lr_scheduler.get_last_lr()[0]
                            else:
                                log_lr = args["lr"]
                            update_progress_bar(progress_bar, epoch, log_loss, log_lr, rank)
                            if args["log_to"] == 'wandb':
                                logger.log({"loss": log_loss, "lr": log_lr}, rank)
                        ddp_loss = torch.zeros(2).to(local_rank)

                    if rank == 0 and args['verbose']:
                        print(f"Batch idx {batch_idx}")

                    prof.step()

                    #Primarily for debugging
                    if args["max_steps"] > 0 and batch_idx > args["max_steps"]:
                        if rank == 0:
                            print("Max steps reached, skipping rest of epoch")
                        break

                # Print + log peak memory usage for the whole fourth step of training
                if epoch == 0 and (rank == 0 or args['verbose']):
                    peak_allocated_memory = torch.cuda.max_memory_allocated(local_rank)
                    peak_reserved_memory  = torch.cuda.max_memory_reserved(local_rank)
                    memory_stats.append(f"Rank {rank}: Peak allocated memory: {peak_allocated_memory/2**30:.2f} GiB")
                    memory_stats.append(f"Rank {rank}: Peak reserved memory:  {peak_reserved_memory/2**30:.2f} GiB")
                    if args["log_to"] == 'wandb':
                        logger.log({"memory/allocated_peak": peak_allocated_memory}, rank)
                        logger.log({"memory/reserved_peak": peak_reserved_memory}, rank)
                
                # Save model - ref: https://github.com/pytorch/pytorch/issues/98823
                # HQQLinear custom state_dict() method causes issues when saving.
                # Model is saved fine when `state_dict()` method is removed.
                # Non param/buffer types are not saved with FSDP.
                # It might be better to just save the trained lora layers.
                # summon_full_params on lora layers and save.
                if args["save_model"]:
                    if rank == 0:
                        os.makedirs(args["output_dir"], exist_ok=True)
                    dist.barrier()
                    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    if args["train_type"] in ["custom_lora", "custom_qlora", "hlora"]:
                        cpu_state_dict = {}

                        trainable_fsdp_modules = [(n,m) for n,m in model.named_modules() if n.endswith(('lora_AB', 'magnitude_layer'))]
                        for prefix, module in trainable_fsdp_modules:
                            prefix = (prefix.replace("_fsdp_wrapped_module.", "")
                                            .replace("_checkpoint_wrapped_module.", "")
                                            .replace("_offload_wrapped_module.", ""))
                            if args['verbose']: print(f"Saving {prefix}")
                            with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, save_policy):
                                cpu_state_dict = {**cpu_state_dict, **{f"{prefix}.{k}":v for k,v in module.state_dict().items()}}
                            dist.barrier()
                            torch.cuda.synchronize()
                        if rank==0:
                            print("Saving trained LoRA weights.")
                            save_file(cpu_state_dict, os.path.join(args["output_dir"], "model_state_dict.safetensors"))
                            print("Done", rank)
                    else:
                        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                            cpu_state_dict = model.state_dict()
                            if rank==0:
                                attribute = dataloader.dataset.attribute
                                print("Saving full model weights.")
                                save_file(cpu_state_dict, os.path.join(args["output_dir"], f"model_state_dict_{epoch}_{attribute}.safetensors"))

        # Synchronize at the end and record time
        init_end_event.record()
        dist.barrier()
        torch.cuda.synchronize()

    if rank == 0:
        print("Finished training", rank)

    # Print time, model, & memory stats
    time_taken = init_start_event.elapsed_time(init_end_event) / 1000
    dist.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        print(f"CUDA event elapsed time: {time_taken} sec")
        logger.log({"time_taken": time_taken}, rank)
    for line in memory_stats:
        print(line)


# Utilities related to model loading
def replace_linear(model:nn.Module, linear_replacement:nn.Module, quant_config:Optional[Dict]=None,
                   skip_modules:List[str]=["lm_head"], **kwargs):
    """
    Replace linear modules with a new Linear module.
    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        linear_replacement (`torch.nn.Module`):
            The linear module that replaces the old one. Only expects standard arguments.
            If other arguments need to be passed, use a lambda.
        skip_modules (`List[str]`, *optional*, defaults to `lm_head`):
            List of modules names not to convert. Defaults to `lm_head`.
    """
    for name, module in model.named_children():
        if name in skip_modules:
            continue

        if len(list(module.children())) > 0:
            replace_linear(module, linear_replacement, quant_config, skip_modules, **kwargs)

        if isinstance(module, torch.nn.Linear):
            if issubclass(linear_replacement, Linear4bit):
                model._modules[name] = linear_replacement(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    **kwargs
                )
            else:
                raise ValueError(f"Unsupported linear replacement: {type(linear_replacement)}")
    return model







# DATASET + DATALOADERS (modified from llama recipes)

# And to get the dataloader
def get_dataloader(tokenizer:PreTrainedTokenizerFast, args:Dict, attribute = "length"):
    """Creates a dataset and appropriate dataloader with distributed sampler."""
    # Importing here rather than at the start to avoid multiprocessing issues
    from datasets import Dataset, load_dataset

    # Load the source dataset
    if args["dataset"] == "alpaca":
        dataset = load_dataset("yahma/alpaca-cleaned")['train']
    elif args['dataset'] == "macsum":
        model_name = "llama31"
        if "mistral" in args['model_name']:
            model_name = "mistral"
        dataset = MACSUM(args['macsum_path'], attribute, tokenizer, mode = 'train', size = args['dataset_samples'], model_type= model_name)


    # truncate dataset so it's evenly divisible by grad_accumulation_steps
    if args['dataset'] == "alpace":
        dataset = dataset.select(range(32))
        dataset = dataset.select(range(0, len(dataset)-len(dataset)%(args["batch_size"]*args["gradient_accumulation_steps"])))

    # Collate function
    def collate_fn(batch, with_attention_mask=False):
        # To list of tensors
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        attention_masks = [torch.tensor(item['attention_mask']) for item in batch]
        labels = [torch.tensor(item['labels']) for item in batch]
        # Pad + truncate
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)[:, :args["context_length"]]
        if with_attention_mask:
            attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)[:, :args["context_length"]]
        else:
            attention_masks = None
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)[:, :args["context_length"]]
        # Return dict
        return {'input_ids': input_ids, 'attention_mask': attention_masks, 'labels': labels}

    # For distributed training, use DistributedSampler
    sampler = DistributedSampler(dataset, seed=args["seed"])

    # Use the custom collate function in DataLoader
    dataloader = DataLoader(dataset, batch_size=args["batch_size"], collate_fn=collate_fn, sampler=sampler)

    return dataloader


def load_and_quantize(module:nn.Module, name:str, value:Tensor, device:torch.device=None, dtype:torch.dtype=None,
                      skip_names:list[str]=[], to_cpu:bool=False, to_meta:bool=False, verbose:bool=False, quant_method:str='bnb',
                      is_dora:bool=False):
    """
    Loads `value` tensor into submodule of `module`, optionally skipping `skip_names` and converting to `dtype`.

    Quantizes `Params4bit` on `device` then places on "cpu" if to_cpu=True or "meta" if to_meta=True.
    """
    def place_on_device(value):
        if to_meta:
            device = 'meta'
        elif to_cpu:
            device = 'cpu'
        return value.to(device=device, dtype=dtype)

    if any([skip_name in name for skip_name in skip_names]):
        if verbose:
            print(f"Skipping {name} because it is in skip_names")
        return

    module_key, _, value_key = name.rpartition('.')
    try:
        submodule = module.get_submodule(module_key)
    except AttributeError as e:
        print(f"Module {module_key} not found:\n{e}")
        return

    try:
        if quant_method=='bnb':
            param = submodule.get_parameter(value_key)
            if isinstance(param, Params4bit):
                # With `sync_module_states=True`, a meta device Params4bit needs to be the same
                # shape as the quantized Params4bit with an initialized quant_state. However,
                # FSDP only syncs parameters and buffers, so the quant_state isn't copied. This
                # workaround quantizes Params4bit to initialize quant_state on all ranks, then
                # replaces Params4bit's data with a meta tensor to free memory on non-rank 0.
                value = type(param)(value.to(device=device, dtype=dtype).data, **param.__dict__).cuda(device)
                if to_meta:
                    value = type(param)(value.data.to("meta"), **value.__dict__)
                elif to_cpu:
                    value = type(param)(value.data.to("cpu"), **value.__dict__)
            else:
                value = type(param)(place_on_device(value).data)
            #print(device, "not here", value.device)


    except AttributeError:
        # it's a buffer
        print("attribute error")
        value = place_on_device(value, device)
        pass
    setattr(submodule, value_key, value)




# Main function, run on each process
def fsdp_main(local_rank:int, world_size:int, args:Dict):

    # Setup and initialize the process group
    os.environ['MASTER_ADDR'] = args["master_addr"]
    os.environ['MASTER_PORT'] = args["master_port"]
    if 'SLURM_PROCID' in os.environ:
        # assumes same number of GPUs per node.
        rank = int(os.environ['SLURM_PROCID']) * torch.cuda.device_count() + local_rank
    else:
        rank = local_rank

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    if args["use_cpu_offload"]:
        torch.set_num_threads(os.cpu_count()//(min(world_size, torch.cuda.device_count())))

    # Start logging
    logger = Logger(args, log_to=args["log_to"], project_name=args["project_name"],
                    entity=args["entity"], group=args["group"], name=args["name"], rank=rank)

    # Timing stuff
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # model precision, qlora compute precison, and FSDP mixed precision policy.
    # The Linear4Bit quant_storage dtype should always match the FSDP param_dtype. The compute_dtype should match the AMP compute dtype.
    # MixedPrecision(param_dtype=fp32, reduce_dtype=fp32, buffer_dtype=fp32) uses `torch.amp.autocast` to control precision.
    # limited qlora testing shows that fp16 only works with autocast while bf16 trains with both pure and autocast modes.
    # TODO: test how often this holds for mp_fp16
    mp_policy = None
    load_param_skip_names = []
    if args["precision"] == "bf16":
        torch_dtype, compute_dtype = torch.bfloat16, torch.bfloat16
    elif args["precision"] == "fp32":
        torch_dtype, compute_dtype = torch.float32, torch.float16
    elif args["precision"] == "fp16_autocast":
        compute_dtype, torch_dtype = torch.float16, torch.float32
        mp_policy = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    elif args["precision"] == "bf16_autocast":
        compute_dtype, torch_dtype = torch.bfloat16, torch.float32
        mp_policy = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    elif args["precision"] == "bf16_buffers_autocast":
        compute_dtype, torch_dtype = torch.bfloat16, torch.bfloat16
        mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.float32)
        load_param_skip_names = ['inv_freq']
    else:
        raise ValueError("Invalid precision")


    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
    tokenizer.pad_token_id = tokenizer.eos_token_id # TODO check if it exists first

    # Set up dataloader

    dataloader = get_dataloader(tokenizer, args, args['attribute_1'])
    if rank == 0 and args['verbose']:
        print("dataset chosen is : ", args['dataset'])
        example = next(iter(dataloader))
        print("example input ")
        print(tokenizer.decode(example['input_ids'][0]))
        print(example['labels'])



    # Create model
    cfg = None
    attn_impl = "sdpa" # torch 2.2 sdpa uses flash attn 2
    if rank == 0 or args['verbose']:
        print("Creating model", rank)

    cfg = AutoConfig.from_pretrained(args["model_name"])
    cfg.use_cache = False
    cfg._attn_implementation = attn_impl
    skip_modules = ["lm_head"]

    # load model on meta device without calling init and replace nn.Linear with Linear4bit
    # quantization
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg)
        model.model = replace_linear(model.model, Linear4bit, compute_dtype=compute_dtype,
                                            quant_type='nf4', quant_storage=torch_dtype, skip_modules=skip_modules)

    model.is_loaded_in_4bit = True
    # Grab the safetensors files that hold the weights
    try:
        idx = hub.cached_file(args["model_name"], SAFE_WEIGHTS_INDEX_NAME)
        files, _ = hub.get_checkpoint_shard_files(args["model_name"], idx)
    except OSError:
        try:
            # This means the model doesn't have a model.safetensors.index.json because it is not sharded
            files = []
            files.append(hub.cached_file(args["model_name"], SAFE_WEIGHTS_NAME))
        except OSError as e:
            # This means the model probably doesn't have a safetensors file
            raise e

    # Load in the weights, using our custom load_and_quantize method which quantizes Params4bit on the fly
    # and then places each layer on CPU or meta if using low_memory to minimize GPU memory usage
    def load_and_quantize_parallel(name_param, model, **kwargs):
        name, param = name_param
        load_and_quantize(model, name, param, **kwargs)

    quant_method = "bnb"
    param_count = sum((p.numel() for n,p in model.named_parameters()))
    if rank == 0 or args['verbose']:
        print("Loading model", rank)
    if rank == 0 and args['verbose']:
        print(f"Total model params: {param_count}")

    n_workers = n_loading_workers(quant_method, param_count) if args["loading_workers"]==-1 else args["loading_workers"]
    if rank == 0 and args['verbose']:
        print(f"Using n_workers: {n_workers} for loading")

    start = time.time()
    #quantizes the checkpoint weights on the fly 
    # so the pipeline is first load the model, make it linear4bit and then quantize the weight and init
    for filename in files:
        weights = safetensors.torch.load_file(filename)
        parallel(load_and_quantize_parallel, iter(weights.items()), n_workers=n_workers, threadpool=True,
                    model=model, dtype=torch_dtype, device=local_rank, skip_names=load_param_skip_names,
                    to_cpu=(args["low_memory"] and rank==0), to_meta=(args["low_memory"] and rank!=0),
                    verbose=args["verbose"], quant_method=quant_method, is_dora=False)
        if rank == 0:
            print(f"Loaded {filename} in {time.time()-start:.3f} seconds")
    #print the device of the params to check if they are on cpu or meta


    if rank == 0 and args["verbose"]:
        print(f"Loaded model weights in {time.time()-start:.3f} seconds")
    # cleanup any extra memory usage from parallel loading
    torch.cuda.empty_cache()
    if rank == 0 or args['verbose']:
        print(f"Rank {rank}: Model created: {torch.cuda.memory_reserved(local_rank)/2**30:.3f} GiB")
        print("initialising hlora")
    hlora_config = HLORAConfig(args['lora_rank_1'], args['lora_rank_2'], args['lora_alpha_1'], args['lora_alpha_2'], args['lora_dropout'], args['lora_target_modules'])
    model = replace_linear4bit_with_hlora(model, hlora_config)
    model = HLORAPeftModel(model, hlora_config)
    #set everything to gpu 0 for now to do quantization
    #model = model.to(local_rank)
    model.set_train_adapters(level_1=True, level_2=False)
    model.set_inference_adapters(level_1=True, level_2=False)
    model.set_gradients()
    # if rank == 0:
    #     print("setup hlora done")
    #     print_model_details(model)



    if args["log_to"] == 'wandb':
        logger.log({"memory/allocated_after_model_created": torch.cuda.memory_allocated(local_rank)}, rank)
        logger.log({"memory/reserved_after_model_creation": torch.cuda.memory_reserved(local_rank)}, rank)


    # Wrap model with llama-recipies or custom LoRA policy
    my_auto_wrap_policy = get_wrapping_policy(custom_policy=args["train_type"] in ["hlora"],
                                                vanilla_policy=args["train_type"] in ["full"])

    if rank == 0 or args['verbose']:
        print("Wrapping model w/ FSDP", rank)

    if args["sharding_strategy"] == "full_shard":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif args["sharding_strategy"] == "shard_grad_op":
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    elif args["sharding_strategy"] == "ddp":
        sharding_strategy = ShardingStrategy.NO_SHARD
    elif args["sharding_strategy"] == "hybrid_full_shard":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif args["sharding_strategy"] == "hybrid_shard_grad_op":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
    else:
        raise ValueError("Invalid FSDP sharding strategy")

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=my_auto_wrap_policy,
        # backward_prefetch=None, #BackwardPrefetch.BACKWARD_PRE
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=True) if args["use_cpu_offload"] else None,
        limit_all_gathers=True, # See https://github.com/pytorch/pytorch/issues/91165
        device_id=torch.cuda.current_device(),
        sync_module_states=args["low_memory"],
        param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if (rank!=0 and args["low_memory"]) else None, # TODO note about meta device and why we need this
        mixed_precision=mp_policy,
    )
    unwrapped_model = model._fsdp_wrapped_module

    if rank == 0 or args['verbose']:
        print(model)
        print_model_details(model)
    if rank == 0 or args['verbose']:
        print(f"Rank {rank}: Wrapped model: {torch.cuda.memory_reserved(local_rank)/2**30:.3f} GiB")
    if args["log_to"] == 'wandb':
        logger.log({"memory/allocated_after_model_wrap": torch.cuda.memory_allocated(local_rank)}, rank)
        logger.log({"memory/reserved_after_model_wrap": torch.cuda.memory_reserved(local_rank)}, rank)

    exit()


    # Synchronize at the start
    dist.barrier()

    # Apply activation checkpointing
    if args["use_gradient_checkpointing"]:
        if args['reentrant_checkpointing']:
            model.enable_input_require_grads()
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.REENTRANT if args['reentrant_checkpointing'] else CheckpointImpl.NO_REENTRANT,

        )

        check_fn = lambda submodule: isinstance(submodule, (LlamaDecoderLayer, MistralDecoderLayer))
        if rank == 0 or args['verbose']:
            print("Applying activation checkpointing", rank)
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )

    if args["use_activation_cpu_offload"]:
        if rank == 0 or args['verbose']:
            print("Applying activation offloading", rank)
        model = offload_wrapper(model)

    if rank == 0 and args['verbose']:
        print("Config:")
        print(cfg)
        print("Model:")
        print(model)
        print("Starting training")
    if rank == 0:
        print("starting training for first attribute : ", args['attribute_1'])
    #train(model, dataloader, args, logger, rank)
    if rank == 0:
        print("done training for the first attribute")
        print("now starting training for the second attribute : ", args['attribute_2'])


    # now the second set of attributes 
    dataloader = get_dataloader(tokenizer, args, args['attribute_2'])
    if rank == 0 and args['verbose']:
        print("dataset chosen is : ", args['dataset'])
        example = next(iter(dataloader))
        print("example input ")
        print(tokenizer.decode(example['input_ids'][0]))
        print(example['labels'])
    #set the gradients for the second level
    unwrapped_model.set_train_adapters(level_1=False, level_2=True)
    unwrapped_model.set_inference_adapters(level_1=True, level_2=True)
    unwrapped_model.set_gradients()
    #train(model, dataloader, args, logger, rank)

    if rank == 0:
        print(model)
        print_model_details(model)


    

    # End logging
    logger.finish(rank=rank)

        

    dist.barrier() # Stop other processes ending while model saving - probably not needed?

    # Clean up
    dist.destroy_process_group()

def validate_args(args):
    if args["n_bits"] != 4 and args["train_type"] not in ["hqq_lora", "hqq_dora", "hqq_llama_pro"]:
        raise ValueError(f"train_type={args['train_type']} doesn't support n_bits={args['n_bits']}. Either don't pass n_bits (to use default of 4) or use any of the hqq training types.")

# Main entry point, validate and package arguments, then call fsdp_main. Scripts importing train.py as an import can invoke this function (invoking main() directly leads to call_parse() overwriting arguments).
def fsdp_qlora(
    world_size: int = -1, # Number of GPUs to use. -1 = all available GPUs.
    train_type: str = "hlora", # "full", "lora", "qlora", or "custom_qlora"
    llama_pro_path: str = None, # Path to the quantized llama pro model
    batch_size: int = 1, # Batch size per GPU. Effective BS = batch_size * world_size * gradient_accumulation_steps
    context_length: int = 512, # Max length of input sequence (in tokens)
    gradient_accumulation_steps: int = 1, # How many steps to accumulate gradients over (increases effective batch size)
    num_epochs: int = 1, # How many epochs of training to do
    dataset: str = "alpaca_sample", # alpaca, alpaca_sample (for a 128-sample test) or "dummy" for 16 long dummy samples
    macsum_path : str = "/home2/tathagato/summarization/MACSUM/dataset/macdoc/train_dataset.json",
    dataset_samples: int = 512, # Number of samples in an epoch if using "alpaca_sample" or "dummy" dataset
    sharding_strategy: str = "full_shard", # Sharding strategy for FSDP
    use_gradient_checkpointing: bool = True, # Use FSDP's activation checkpointing
    reentrant_checkpointing: bool = False, # Use re-entrant autograd activation checkpointing. Setting to True can use less GPU memory with BNB QLoRA
    use_cpu_offload: bool = True, # Use FSDP's CPU offloading
    use_activation_cpu_offload: bool = False, # Use FSDP's activation CPU offloading
    low_memory: bool = True, # Load one copy of the model into CPU memory before sharding with FSDP. For QLoRA, quantizes each layer individually on GPU before placing on CPU.
    no_sync: bool = False, # Prevent gradient sync until update step. Likely uses more memory. Required for `use_cpu_offload` and `gradient_accumulation_steps > 1`
    precision: str = "bf16", # Training precision. autocast precisions use mixed precision
    model_name: str = "meta-llama/Llama-2-7b-hf", # Which model to train - e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    save_model: bool = False, # Save the resulting model
    output_dir: str = "output", # Output directory to save the final model to
    lora_rank_1: int = 64, # LoRA rank for lora/qlora
    lora_alpha_1: int = 16, # LoRA alpha for lora/qlora
    lora_rank_2: int = 32,  #hlora dimensions for second layer
    lora_alpha_2: int = 16, # hlora alpha for the second layer
    lora_dropout: float = 0.1, # LoRA dropout for lora/qlora
    lora_target_modules: str = "all", # If 'default', uses peft defaults. Use 'all' for our best guess for Llama models
    verbose: bool = False, # Whether to print extra info for debugging
    lr: float = 1e-5, # Learning rate
    apply_gradient_clipping: bool = False, # Apply gradient norm clipping
    grad_norm: float = 0.3, # Gradient norm clipping
    wd: float = 0.1, # Weight decay
    profile_memory: bool = False, # Profile memory usage for the first few batches. Keep false for training. May increase memory usage.
    optimizer: str = "adamw", # Optimizer. PyTorch 2.4 nightly adds CPU fused Adam/AdamW which should improve offload training speed.
    lr_scheduler: str = "constant", # Learning Rate Scheduler. linear and cosine warm up for 10% of training steps.
    loading_workers: int = -1, # Number of layers to load and quantize in parallel per GPU. Default of -1 uses heuristics to set worker count.
    log_to: str = "tqdm", # Where to log output
    master_addr: str = "localhost", # For distributed training
    master_port: str = "12355", # For distributed training, must be the same for all processes
    seed: int = 42, # Random seed
    project_name: str = "fsdp_qlora", # For wandb logging
    name: str = None, # For wandb logging
    group: str = None, # For wandb logging
    entity: str = None, # For wandb logging
    n_bits: int = 4, # passed to hqq
    #Profiling args
    profile: bool_arg = False, # Whether to profile with torch.profiler
    profiling_output: str = "profiles", # Output file prefix for profiling
    overwrite_profiling_output: bool = True, # Overwrite output directory
    with_stack: bool_arg = False, # Output stacks for profiling. Note that setting export_memory_timeline will automatically export traces since `with_stack` must be true to profile memory.
    with_shapes: bool_arg = False, # Output shapes for profiling. Can impact performance.  Note that setting export_memory_timeline will automatically export traces since `with_shapes` must be true to profile memory.
    export_trace: bool_arg = True, # Output trace for profiling
    export_memory_timeline: bool_arg = False, # Output memory timelinefor profiling
    wait_steps: int = 1, # Wait steps when running profiler.  Only used if repeat != 0.
    warmup_steps: int = 1, # Warmup steps when running profiler
    active_steps: int = 2,  # Active steps when running profiler
    repeat: int = 0, #Number of profiler cycles (wait + warmup + active) if > 0, else repeats forever
    profiling_frequency: int = 10, # Profiling frequency in steps.  Only used if repeat == 0, in which case wait_steps will be set to profiling_frequency - (warmup_steps + active_steps) such that the effective cycle length == profiling_frequency
    max_steps: int = -1, # Max number of training steps (in units of batches) per epoch. -1 means no max_steps, otherwise training loop breaks after `max_steps` each epoch.\
    attribute_1 : str = "length",
    attribute_2 : str = "extractiveness"
):  
    """
    Train a model with FSDP and QLoRA/QDoRA.

    Args:

        world_size: Number of GPUs to use. -1 = all available GPUs.
        train_type: "full", "lora", "qlora", or "custom_qlora"
        llama_pro_path: Path to the quantized llama pro model
        batch_size: Batch size per GPU. Effective BS = batch_size * world_size * gradient_accumulation_steps
        context_length: Max length of input sequence (in tokens)
        gradient_accumulation_steps: How many steps to accumulate gradients over (increases effective batch size)
        num_epochs: How many epochs of training to do
        dataset: alpaca, alpaca_sample (for a 128-sample test) or "dummy" for 16 long dummy samples
        dataset_samples: Number of samples in an epoch if using "alpaca_sample" or "dummy" dataset
        sharding_strategy: Sharding strategy for FSDP
        use_gradient_checkpointing: Use FSDP's activation checkpointing
        reentrant_checkpointing: Use re-entrant autograd activation checkpointing. Setting to True can use less GPU memory with BNB QLoRA
        use_cpu_offload: Use FSDP's CPU offloading
        use_activation_cpu_offload: Use FSDP's activation CPU offloading
        low_memory: Load one copy of the model into CPU memory before sharding with FSDP. For QLoRA, quantizes each layer individually on GPU before placing on CPU.
        no_sync: Prevent gradient sync until update step. Likely uses more memory. Required for `use_cpu_offload` and `gradient_accumulation_steps > 1`
        precision: Training precision. autocast precisions use mixed precision
        model_name: Which model to train - e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        save_model: Save the resulting model
        output_dir: Output directory to save the final model to
        lora_rank: LoRA rank for lora/qlora
        lora_alpha: LoRA alpha for lora/qlora
        lora_dropout: LoRA dropout for lora/qlora
        lora_target_modules: If 'default', uses peft defaults. Use 'all' for our best guess for Llama models
        verbose: Whether to print extra info for debugging
        lr: Learning rate
        apply_gradient_clipping: Apply gradient norm clipping
        grad_norm: Gradient norm clipping
        wd: Weight decay
        profile_memory: Profile memory usage for the first few batches. Keep false for training. May increase memory usage.
        optimizer: Optimizer. PyTorch 2.4 nightly adds CPU fused Adam/AdamW which should improve offload training speed.
        lr_scheduler: Learning Rate Scheduler. linear and cosine warm up for 10% of training steps.
        loading_workers: Number of layers to load and quantize in parallel per GPU. Default of -1 uses heuristics to set worker count.
        log_to: Where to log output
        master_addr: For distributed training
        master_port: For distributed training, must be the same for all processes
        seed: Random seed
        project_name: For wandb logging
        name: For wandb logging
        group: For wandb logging
        entity: For wandb logging
        n_bits: passed to hqq
        profiling_output: Output file for profiling
    """

    # Set world size
    if world_size == -1:
        world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")

    # Get all args which will be passed to fsdp_main
    args = dict(locals())
    set_seed(args['seed'])
    validate_args(args)
    if args['verbose']: print(args)

    # If lora_target_modules is 'all', set sensible defaults for llama + mistral type modules
    # See peft.utils.constants -> TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING for the current defaults
    if lora_target_modules == "all":
        args["lora_target_modules"] = ["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
    elif lora_target_modules.lower() == "default":
        args["lora_target_modules"] = None

    if args["precision"] in ["bf16", "bf16_autocast", "bf16_buffers_autocast"] and not torch.cuda.is_bf16_supported():
        raise ValueError('Current device does not support bfloat16')

    # Set no_sync if using cpu_offload and gradient accumulation. Turn off if not using gradient accumulation
    if args["use_cpu_offload"] and args["gradient_accumulation_steps"] > 1:
        args["no_sync"] = True
    elif args["no_sync"] and args["gradient_accumulation_steps"] == 1:
        args["no_sync"] = False



    if args["optimizer"] in ["fused_adam", "fused_adamw"] and args["use_cpu_offload"] and parse(torch.__version__) < parse("2.4dev"):
        raise ValueError(f"Optimizer '{args['optimizer']}' with `use_cpu_offload=True` requires at least PyTorch 2.4 Nightly with fused Adam/AdamW CPU support.")

    # Run
    mp.spawn(fsdp_main,
        args=(world_size, args),
        nprocs=torch.cuda.device_count(),
        join=True)

# Entry point, one line wrapper around fsdp_qlora(), use fastcore's call_parse to parse args from command line
@call_parse()
def main(
    world_size: int = -1, # Number of GPUs to use. -1 = all available GPUs.
    train_type: Param("", choices=["full", "lora", "qlora", "custom_qlora", "custom_lora", "hqq_lora", "hqq_dora", "bnb_dora", "bnb_llama_pro", "hqq_llama_pro", "hlora"]) = "hlora", # "full", "lora", "qlora", or "custom_qlora"
    llama_pro_path: str = None, # Path to the quantized llama pro model
    batch_size: int = 1, # Batch size per GPU. Effective BS = batch_size * world_size * gradient_accumulation_steps
    context_length: int = 512, # Max length of input sequence (in tokens)
    gradient_accumulation_steps: int = 1, # How many steps to accumulate gradients over (increases effective batch size)
    num_epochs: int = 1, # How many epochs of training to do
    dataset: Param("", choices=["alpaca", "alpaca_sample", "dummy", "guanaco", "sql", "orca_math","macsum"]) = "alpaca_sample", # alpaca, alpaca_sample (for a 128-sample test) or "dummy" for 16 long dummy samples
    macsum_path : str = "/home2/tathagato/summarization/MACSUM/dataset/macdoc/train_dataset.json",
    dataset_samples: int = 512, # Number of samples in an epoch if using "alpaca_sample" or "dummy" dataset
    sharding_strategy: Param("", choices=["full_shard", "shard_grad_op", "ddp", "hybrid_full_shard", "hybrid_shard_grad_op"]) = "full_shard", # Sharding strategy for FSDP
    use_gradient_checkpointing: bool_arg = True, # Use FSDP's activation checkpointing
    reentrant_checkpointing: bool_arg = False, # Use re-entrant autograd activation checkpointing. Setting to True can use less GPU memory with BNB QLoRA
    use_cpu_offload: bool_arg = True, # Use FSDP's CPU offloading
    use_activation_cpu_offload: bool_arg = False, # Use FSDP's activation CPU offloading
    low_memory: bool_arg = True, # Load one copy of the model into CPU memory before sharding with FSDP. For QLoRA, quantizes each layer individually on GPU before placing on CPU.
    no_sync: bool_arg = False, # Prevent gradient sync until update step. Likely uses more memory. Required for `use_cpu_offload` and `gradient_accumulation_steps > 1`
    precision: Param("", choices=["fp32", "bf16", "fp16_autocast", "bf16_autocast", "bf16_buffers_autocast"]) = "bf16", # Training precision. autocast precisions use mixed precision
    model_name: str = "meta-llama/Llama-2-7b-hf", # Which model to train - e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    save_model: bool_arg = False, # Save the resulting model
    output_dir: str = "output", # Output directory to save the final model to
    lora_rank_1: int = 64, # LoRA rank for lora/qlora
    lora_alpha_1: int = 16, # LoRA alpha for lora/qlora
    lora_rank_2: int = 32, 
    lora_alpha_2: int = 16,
    lora_dropout: float = 0.1, # LoRA dropout for lora/qlora
    lora_target_modules: Param("", choices=["all", "default"]) = "all", # If 'default', uses peft defaults. Use 'all' for our best guess for Llama models
    verbose: bool_arg = False, # Whether to print extra info for debugging
    lr: float = 1e-5, # Learning rate
    apply_gradient_clipping: bool_arg = False, # Apply gradient norm clipping
    grad_norm: float = 0.3, # Gradient norm clipping
    wd: float = 0.1, # Weight decay
    profile_memory: bool_arg = False, # Profile memory usage for the first few batches. Keep false for training. May increase memory usage.
    optimizer: Param("", choices=["adamw", "adam", "sgd", "adadelta"]) = "adamw", # Optimizer
    lr_scheduler: Param("", choices=["constant", "linear", "cosine"]) = "constant", # Learning Rate Scheduler. linear and cosine warm up for 10% of training steps.
    loading_workers: int = -1, # Number of layers to load and quantize in parallel per GPU. Default of -1 uses heuristics to set worker count.
    log_to: Param("", choices=["tqdm", "wandb", "stdout"]) = "tqdm", # Where to log output
    master_addr: str = "localhost", # For distributed training
    master_port: str = "12355", # For distributed training, must be the same for all processes
    seed: int = 42, # Random seed
    project_name: str = "fsdp_qlora", # For wandb logging
    name: str = None, # For wandb logging
    group: str = None, # For wandb logging
    entity: str = None, # For wandb logging
    n_bits: int = 4, # passed to hqq
    profile: bool_arg = False, # Whether to profile with torch.profiler
    profiling_output: str = "profiles", # Output file prefix for profiling
    with_stack: bool_arg = False, # Output stacks for profiling. Note that setting export_memory_timeline will automatically export traces since `with_stack` must be true to profile memory.
    with_shapes: bool_arg = False, # Output shapes for profiling. Can impact performance.  Note that setting export_memory_timeline will automatically export traces since `with_shapes` must be true to profile memory.
    export_trace: bool_arg = True, # Output trace for profiling
    export_memory_timeline: bool_arg = False, # Output memory timelinefor profiling
    wait_steps: int = 0, # Wait steps when running profiler.  Only used if repeat != 0.
    warmup_steps: int = 1, # Warmup steps when running profiler
    active_steps: int = 2,  # Active steps when running profiler
    repeat: int = 0, #Number of profiler cycles (wait + warmup + active) if > 0, else repeats forever
    profiling_frequency: int = 10, # Profiling frequency in steps.  Only used if repeat == 0, in which case wait_steps will be set to profiling_frequency - (warmup_steps + active_steps) such that the effective cycle length == profiling_frequency
    max_steps: int = -1, # Max number of training steps (in units of batches) per epoch. -1 means no max_steps, otherwise training loop breaks after `max_steps` each epoch.
    attribute_1 : str = "length",
    attribute_2 : str = "extractiveness"

):
    fsdp_qlora(**locals())