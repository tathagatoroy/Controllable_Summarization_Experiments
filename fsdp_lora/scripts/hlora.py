import torch
import torch.nn as nn
from peft import PeftConfig, PeftModel,LoraConfig
from peft.tuners.lora import LoraLayer
from peft.utils import PeftType
from transformers.utils import PushToHubMixin
from bitsandbytes.nn import Linear4bit

class HLORAConfig(LoraConfig):
    def __init__(self, lora_rank_1=32, lora_rank_2=16, lora_alpha_1=16, lora_alpha_2=8, lora_dropout=0.1, target_modules = ["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],**kwargs):
        super().__init__(**kwargs)
        self.peft_type = PeftType.LORA
        self.lora_rank_1 = lora_rank_1
        self.lora_rank_2 = lora_rank_2
        self.lora_alpha_1 = lora_alpha_1
        self.lora_alpha_2 = lora_alpha_2
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules

class HLORA(nn.Module):
    def __init__(self, base_layer, config):
        super().__init__()

        self.config = config
        self.base_layer = base_layer
        
        dtype = getattr(base_layer, "compute_dtype", next(base_layer.parameters()).dtype)
        device = next(base_layer.parameters()).device

        lora_A = nn.Linear(base_layer.in_features, config.lora_rank_1, bias=False, device=device, dtype=dtype)
        lora_B = nn.Linear(config.lora_rank_1, base_layer.out_features, bias=False, device=device, dtype=dtype)
        lora_B.weight.data.zero_()

        lora_C = nn.Linear(base_layer.in_features, config.lora_rank_2, bias=False, device=device, dtype=dtype)
        lora_D = nn.Linear(config.lora_rank_2, config.lora_rank_1, bias=False, device=device, dtype=dtype)
        lora_E = nn.Linear(config.lora_rank_1, config.lora_rank_2, bias=False, device=device, dtype=dtype)
        lora_F = nn.Linear(config.lora_rank_2, base_layer.out_features, bias=False, device=device, dtype=dtype)
        lora_F.weight.data.zero_()
        lora_D.weight.data.zero_()

        self.lora_A1B1 = nn.Sequential(
            lora_C, lora_D,lora_E, lora_F
        )
        self.lora_AB = nn.Sequential(lora_A, lora_B)

        self.scaling_1 = self.config.lora_alpha_1 / self.config.lora_rank_1
        self.scaling_2 = self.config.lora_alpha_2 / self.config.lora_rank_2
        self.lora_dropout = nn.Dropout(config.lora_dropout)

        self.do_inference = [True, False]
        self.do_train = [True, False]

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        result = result.clone()

        #self.lora_AB.requires_grad = self.do_train[0]
        #self.lora_A1B1.requires_grad = self.do_train[1]

        requires_conversion = not torch.is_autocast_enabled()

        if self.do_inference[0]:
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(next(iter(self.lora_AB)).weight.dtype)
            output = self.lora_AB(self.lora_dropout(x))
            if requires_conversion:
                output = output.to(expected_dtype)
            output = output * self.scaling_1
            result += output

        if self.do_inference[1]:
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(next(iter(self.lora_A1B1)).weight.dtype)
            output = self.lora_A1B1(self.lora_dropout(x))
            if requires_conversion:
                output = output.to(expected_dtype)
            output = output * self.scaling_2
            result += output

        return result

class HLORAPeftModel(nn.Module):
    #init nn.Module

    def __init__(self, model, peft_config, adapter_name="default"):
        nn.Module.__init__(self)  # Correctly initialize nn.Module #init nn.Module not the others
        self.adapter_name = adapter_name
        self.base_model = model
        self.set_train_adapters(level_1=True, level_2=False)
        self.set_inference_adapters(level_1=True, level_2=False)
        

    def set_train_adapters(self, level_1=True, level_2=False):
        for module in self.modules():
            if isinstance(module, HLORA):
                module.do_train = [level_1, level_2]
        self.do_train = [level_1,level_2]

    def set_inference_adapters(self, level_1=True, level_2=False):
        for module in self.modules():
            if isinstance(module, HLORA):
                module.do_inference = [level_1, level_2]
        self.do_inference = [level_1, level_2]
    def set_gradients(self):
        """ set the gradient of all modules in the model
            base model : requires grad always false
            lora_AB = nn.Sequential(self.lora_A, self.lora_B) req_grad = False if self.lora_AB.do_train[0] is False else true 
            lora_A1B1 = nn.Sequential(lora_A1B1 = nn.Sequential(nn.Sequential(self.lora_C, self.lora_D),nn.Sequential(self.lora_E, self.lora_F)) if self.lora_A1B!.do_train[False]
        """
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Set HLORA module gradients
        for module in self.modules():
            if isinstance(module, HLORA):
                # Set lora_AB gradients
                for param in module.lora_AB.parameters():
                    param.requires_grad = self.do_train[0]

                # Set lora_A1B1 gradients
                for param in module.lora_A1B1.parameters():
                    param.requires_grad = self.do_train[1]




def replace_linear4bit_with_hlora(model, peft_config):
    for name, module in model.named_children():
        if isinstance(module, Linear4bit) and getattr(module, "compute_dtype", None) is not None:
            # Check if it's a Linear4bit
            #print("setting attribute")
            setattr(model, name, HLORA(module, peft_config))
        else:
            replace_linear4bit_with_hlora(module, peft_config)
    return model

# Copyconfig = HLORAConfig(lora_rank_1=32, lora_rank_2=16, lora_alpha_1=16, lora_alpha_2=8, lora_dropout=0.1)
# model = replace_linear4bit_with_hlora(original_model, config)
# peft_model = HLORAPeftModel(model, config)

# # Set which adapters to use for training
# peft_model.set_train_adapters(level_1=True, level_2=False)

# # Set which adapters to use for inference
# peft_model.set_inference_adapters(level_1=True, level_2=True)
# Usage example:
# config = HLORAConfig(lora_rank_1=32, lora_rank_2=16, lora_alpha_1=16, lora_alpha_2=8, lora_dropout=0.1)
# model = replace_linear4bit_with_hlora(original_model, config)
# peft_model = HLORAPeftModel(model, config)
#
# # Set which adapters to use for training
# peft_model.set_train_adapters(level_1=True, level_2=False)
#
# # Set which adapters to use for inference
# peft_model.set_inference_adapters(level_1=True, level_2=True)