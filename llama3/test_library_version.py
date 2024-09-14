import torch 
import transformers
import peft
import trl
import accelerate
import bitsandbytes
import deepspeed
import torchvision
# https://huggingface.co/docs/peft/main/en/accelerate/fsdp#use-peft-qlora-and-fsdp-for-finetuning-large-models-on-multiple-gpus
# , we first need bitsandbytes>=0.43.0, accelerate>=0.28.0, transformers>4.38.2, trl>0.7.11 and peft>0.9.0.
if __name__ == '__main__':
    print('torch version:', torch.__version__)
    print('transformers version:', transformers.__version__)
    print('peft version:', peft.__version__)
    print('trl version:', trl.__version__)
    print('accelerate version:', accelerate.__version__)
    print('bitsandbytes version:', bitsandbytes.__version__)
    print("deepseed version:", deepspeed.__version__)
    print("torchvision version:", torchvision.__version__)