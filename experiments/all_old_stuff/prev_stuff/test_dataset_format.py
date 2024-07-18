#https://github.com/huggingface/trl/pull/444
from datasets import load_dataset
import pprint 
import os
from config import *
pretty_print = pprint.PrettyPrinter(indent=4)
from dataset import create_huggingface_dataset_from_dictionary
from utils import *


def formatting_prompts_func(examples : dict) -> list:
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        response = examples["output"][i]

        text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        
        ### Instruction:
        {instruction}
        
        ### Input:
        {input_text}
        
        ### Response:
        {response}
        '''

        output_text.append(text)
    return output_text


if __name__ == "__main__":
    dataset = load_dataset_from_path(val_dataset_path, 'length')
    print("size of the dataset: ", len(dataset))

    examples = dataset[:2] 
    formatted_examples = formatting_prompts_func(examples)

    for index,example in enumerate(formatted_examples):
        print(f"Example {index}")
        print(example)
        #tokenize the example with large max length 
        tokenized_example = tokenizer(example, max_length = 4024, padding = "max_length", truncation = True)
        #get the length of tokenized example before padding 
        print("length of tokenized example before padding: ", len(tokenized_example["input_ids"]))
        #get the length of tokenized example after padding
        


        print("\n")
        if index == 10:
            break   






