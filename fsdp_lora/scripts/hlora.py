#implementation of hierarchichal lora 
# L(x) = Base(x) + alpa_2AB(x) + alpha_1(CD)(EF)(x)
import torch
import torch.nn as nn

class LORA(nn.Module):
    def __init__(self, base_layer, lora_rank_1 = 32, lora_rank_2 = 16 , lora_alpha_1 = 16 , lora_alpha_2 = 8, lora_dropout):
        super().__init__()
        self.base_layer = base_layer
        dtype = getattr(base_layer, "compute_dtype", next(base_layer.parameters()).dtype)
        device = next(base_layer.parameters()).device
        lora_A = nn.Linear(base_layer.in_features, lora_rank_1, bias=False, device=device, dtype=dtype)
        lora_B = nn.Linear(lora_rank, base_layer.out_features, bias=False, device=device, dtype=dtype)
        lora_B.weight.data.zero_()

        lora_C = nn.Linear(base_layer.in_features, lora_rank_2, bias=False, device=device, dtype=dtype)
        lora_D = nn.Linear(lora_rank_2, lora_rank_1, bias=False, device=device, dtype=dtype)
        lora_E = nn.Linear(lora_rank_1, lora_rank_2, bias=False, device=device, dtype=dtype)
        lora_F = nn.Linear(lora_rank_2, base_layer.out_features, bias=False, device=device, dtype=dtype)

        A1 = nn.Sequential(lora_C, lora_D)
        B1 = nn.Sequential(lora_E, lora_F)
        self.lora_A1B1 = nn.Sequential(self.A1, self.B1)


        self.lora_AB = nn.Sequential(lora_A, lora_B)

        self.lora_alpha_1 = lora_alpha_1 
        self.lora_alpah_2 = lora_alpha_2
        self.rank_1 = rank_1
        self.rank_2 = rank_2 
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.scaling = self.lora_alpha_1 / lora_rank_
        self.do_inference = [True, False] # use the first lora layer for inference, not the second one 
        self.do_train = [True, False] # use the first lora layer for training, not the second one


    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        result = self.base_layer(x, *args, **kwargs)
        # As per Tim Dettmers, for 4bit, we need to defensively clone here.
        # The reason is that in some cases, an error can occur that backprop
        # does not work on a manipulated view. This issue may be solved with
        # newer PyTorch versions but this would need extensive testing to be
        # sure.
        result = result.clone()
        if self.do_train[0]:
            self.lora_AB.requires_grad = True
        else:
            self.lora_AB.requires_grad = False
        if self.do_train[1]:
            self.lora_A1B1.requires_grad = True
        else:
            self.lora_A1B1.requires_grad = False

        if self.do_inference[0] == False and self.do_inference[1] == False:
            print("just base layer")
            return result
        elif self.do_inference[0] == True and self.do_inference[1] == False:
            print("just first layer")

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(next(iter(self.lora_AB)).weight.dtype) #this is the dtype of the weights of the first layer AB

            output = self.lora_AB(self.lora_dropout(x))
            if requires_conversion:
                output = output.to(expected_dtype)
            output = output * self.scaling
            result += output
        
        elif self.do_inference[0] == False and self.do_inference[1] == True:
            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(next(iter(self.lora_A1B1)).weight.dtype)
            output = self.lora_A1B1(self.lora_dropout(x))
            if requires_conversion:
                output = output.to(expected_dtype)
            output = output * self


        return result