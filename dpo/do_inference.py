from safetensors import safe_open
import torch
from transformers import LlamaForCausalLM, BitsAndBytesConfig, AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import os 
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys
import pickle as pkl
from dataset import MACSUM
import tqdm
from peft import PeftModel, PeftConfig
import time
from torch.utils.data import DataLoader
from transformers import pipeline
def collate_fn(batch):
    return {key: [example[key] for example in batch] for key in batch[0].keys()}
def generate_text_with_pipeline(generator, dataset, batch_size=20, max_length=2048, args=None):


    #for debug 
    
    # Create a DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    all_generated_texts = []
    
    for batch in tqdm.tqdm(dataloader):
        # Extract the prompts from the batch
        prompts = batch['prompt']
        for i in range(len(prompts)):
            print(prompts[i])
        
        # Generate text for the batch
        outputs = generator(prompts, do_sample=args.do_sample, top_p=args.top_p, top_k=args.top_k, max_new_tokens=args.max_new_tokens, num_return_sequences=1)
        
        # Extract the generated text from the outputs
        generated_texts = [output[0]['generated_text'] for output in outputs]
        predicted_summaries = [text.split("\n\n")[-1] for text in generated_texts]
        
        # Store the results
        for i, generated_text in enumerate(generated_texts):
            all_generated_texts.append({
                'input': batch['input'][i], #src text 
                'prompt': batch['prompt'][i], #prompt used 
                'generated_text': generated_text, # generated text
                'reference': batch['output'][i], #reference text
                'control_value': batch['control_value'][i], #control value
                'control_attribute': batch['control_attribute'][i] , #control attribute
                'predicted_summary' : predicted_summaries[i] #predicted summary 
                })
    
    return all_generated_texts
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default = "/scratch/tathagato/dpo_macsum_storm_llama_extractiveness/checkpoint-537" )
    parser.add_argument("--model_id", type=str, default= "akjindal53244/Llama-3.1-Storm-8B")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--attribute", type=str, default="extractiveness")
    parser.add_argument("--dataset_size", type=int, default=3)
    parser.add_argument('--do_sample', type=bool, default=True, help="Whether to sample during text generation.")
    parser.add_argument('--top_p', type=float, default=0.95, help="Top-p for nucleus sampling.")
    parser.add_argument('--top_k', type=int, default=50, help="Top-k for top-k sampling.")
    parser.add_argument('--max_new_tokens', type=int, default=300, help="Maximum number of new tokens to generate.")
    parser.add_argument('--num_return_sequences', type=int, default=1, help="Number of sequences to return during generation.")


    args = parser.parse_args()
    args.output_path = os.path.join(args.checkpoint_dir, "inference_results.pkl")
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant = True,
)

    #load the model
    model = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=bnb_config)
    # Initialize the pipeline
    #https://github.com/huggingface/peft/issues/218#issuecomment-1512768373
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    generator = pipeline('text-generation', model=model, tokenizer = tokenizer)
    generator.model = PeftModel.from_pretrained(model = model, model_id = args.checkpoint_dir, is_trainable=True)
    print(generator.model)
    print(generator.model.print_trainable_parameters())
    model_type = "llama31"
    if "mistral" in args.model_id:
        model_type = "mistral"

    dataset = MACSUM(tokenizer = tokenizer, attribute= args.attribute, model_type= model_type, size = -1)
    print(f"Dataset size: {len(dataset)}")

    all_generated_texts = generate_text_with_pipeline(generator, dataset, args=args)
      #print(all_generated_texts)
    args.output_path = os.path.join(args.checkpoint_dir, f"inference_results_model_type_{model_type}_size_{len(dataset)}_attribute_{args.attribute}.pkl")
    with open(args.output_path, "wb") as f:
        pkl.dump(all_generated_texts, f)
    

