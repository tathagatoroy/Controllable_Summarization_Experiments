
## IMPLEMENTATION NOTES:
Implementation notes and details
    * certain hyperparameters will be kept constant for all experiments. Examples :
        * quant_config
        * peft config
        * use_4bit 
        * lora_r = 32
        * lora_alpha = 16
    For each experiment subtype they are stored under global and specific parameters are stored inside the experiment type

    