from datasets import load_dataset , Dataset, concatenate_datasets 
import numpy as np
import pandas as pd
import random
import json
import pprint 
import os
pretty_print = pprint.PrettyPrinter(indent=4)

def formatting_prompts_for_inference(instruction : str, input_text : str) -> str:


    text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input_text}

    ### Response:
    '''

    return text


#some constants 
allowed_controllable_aspects = ['length','extractiveness', 'specificity','topic','Speaker']
def create_prompt(example : dict , controllable_aspect : str) -> str:
    assert controllable_aspect in allowed_controllable_aspects, f"Invalid controllable aspect {controllable_aspect}"
    assert example['control_attribute'][controllable_aspect] != "", f"Control attribute {controllable_aspect} is not present in the example"

    control_description = example['control_attribute'][controllable_aspect]
    src_text = example['source']
    base_prompt = f"Write a summary of the text."
    if controllable_aspect == 'length':
        ca_aspect = f"The summary should be {control_description} in length."
    elif controllable_aspect == 'extractiveness':
        ca_aspect = f"The summary should be {control_description} in extractiveness"
    elif controllable_aspect == 'specificity':
        ca_aspect = f"The summary should be {control_description} in specificity"
    elif controllable_aspect == 'topic':
        ca_aspect = f"The summary should be focussed on the topic {control_description}"
    elif controllable_aspect == 'Speaker':
        ca_aspect = f"The summary should be written from the perspective of {control_description}"
    prompt = f"{base_prompt} {ca_aspect}. The input text is given below \n"
    text = f"{src_text}"
    prompt_for_inference = formatting_prompts_for_inference(prompt, text)
    return {"instruction" : prompt, "input" : text, "output" : example['reference'], "control_attribute" : controllable_aspect, "control_value" : example['control_attribute'][controllable_aspect], "prompt_for_inference" : prompt_for_inference}

def create_multi_attribute_prompts(example : dict, controllable_aspects : list) -> str:
    for control_attribute in controllable_aspects:
        assert control_attribute in allowed_controllable_aspects, f"Invalid controllable aspect {control_attribute}"
        assert example['control_attribute'][control_attribute] != "", f"Control attribute {control_attribute} is not present in the example"
    src_text = example['source']
    base_prompt = f"Write a summary of the text."
    ca_aspects = []
    for control_attribute in controllable_aspects:
        if control_attribute == 'length':
            ca_aspects.append(f"The summary should be {example['control_attribute'][control_attribute]} in length")
        elif control_attribute == 'extractiveness':
            ca_aspects.append(f"The summary should be {example['control_attribute'][control_attribute]} in extractiveness")
        elif control_attribute == 'specificity':
            ca_aspects.append(f"The summary should be {example['control_attribute'][control_attribute]} in specificity")
        elif control_attribute == 'topic':
            ca_aspects.append(f"The summary should be focussed on the topic {example['control_attribute'][control_attribute]}")
        elif control_attribute == 'Speaker':
            ca_aspects.append(f"The summary should be written from the perspective of {example['control_attribute'][control_attribute]}")
    #join the aspects with and
    ca_aspect = " and ".join(ca_aspects)
    prompt = f"{base_prompt} {ca_aspect}. The input text is given below \n"
    text = f"{src_text}"
    prompt_for_inference = formatting_prompts_for_inference(prompt, text)

    return {"instruction" : prompt, "input" : text, "output" : example['reference'], "control_attribute" : controllable_aspects, "control_value" : [example['control_attribute'][control_attribute] for control_attribute in controllable_aspects], "prompt_for_inference" : prompt_for_inference}

def create_huggingface_dataset_from_dictionary(dataset_dict : dict, controllable_aspect : str) -> Dataset:
    examples = []
    for key in dataset_dict.keys():
        example = dataset_dict[key]
        if example['control_attribute'][controllable_aspect] == "":
            continue
        example = create_prompt(example, controllable_aspect)
        examples.append(example)

    keys = examples[0].keys()
    data = {key: [example[key] for example in examples] for key in keys}
    return Dataset.from_dict(data)

def create_multiattribute_dataset_from_dictionary(dataset_dict : dict, controllable_aspects : list) -> Dataset:
    examples = []
    for key in dataset_dict.keys():
        example = dataset_dict[key]
        flag = True
        for control_attribute in controllable_aspects:
            if example['control_attribute'][control_attribute] == "":
                flag = False
                break
        if not flag:
            continue
        example = create_multi_attribute_prompts(example, controllable_aspects)
        examples.append(example)
    keys = examples[0].keys()
    data = {key: [example[key] for example in examples] for key in keys}
    return Dataset.from_dict(data)






    

if __name__=='__main__':
    val_dataset_path = "/home2/tathagato/summarization/MACSum/dataset/macdoc/val_dataset.json"
    val_dataset_dict = json.load(open(val_dataset_path,"r"))
    val_dataset = create_huggingface_dataset_from_dictionary(val_dataset_dict, 'length')
    print(len(val_dataset))
    pprint.pprint(val_dataset[0])
    

    





