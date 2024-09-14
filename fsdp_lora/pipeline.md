this is the pipeline on how the fsdp_main function operates:
    1. setup in environment variables mostly for distributed training
    2. setup up device , node and gpu math
    3. get local rank 
    4. timing utils to setup for I am assuming profiling
    5. setup precision, this is important. as there are certain restrictions in what is allowed
        compute_dtype and torch_dtype is setup
        and compute dtype should match with Linear4Bit quant quant_storage
    6.load tokenizer and use pad as eos_token_id , I am assuming this is somewhat related to packed dataset ?
    7. get dataloader : TODO : see the dataset format 
    8. setup up config and attn_impl , see if torch 2.2 is setup
    9. setup what kind of train type to do. For now qlora later need to run custon\
    10. does some computation of the model layers
    11. load model in meta device 
    12 . replaces the linear4bit layers with qlora
    13. load the safetensors
    14. load and quantize the weight
    15. if normal lora/qlora setup peftconfig and get_peft_model
    16. if custom lora/qlora, replace model linear4bit with lora/qlora layers and set trainainle and non trainaible parameters
    17. get wrapping policy : a partial function which returns the kind of object a layer/module have whether they need to wrap it
    18. set sharding policy : treating it as blackbox now
    19. load fsdp model
    20. set gradient checkpointing
    21 optimizer and lr scheduler and batchsize math
    22. autocast, mixed_precisiom , scaling parameters
