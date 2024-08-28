from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm.auto import tqdm
import json
import torch
import copy

# DATASET + DATALOADERS (modified from llama recipes)
# Formatting prompts in alpaca
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "zero_shot_prompt": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "{instruction}\n\n{input}\n\n Response:"
    ),
}

# Dataset class
class MACSUM(Dataset):
    def __init__(self, dataset_path = "/home2/tathagato/summarization/MACSUM/dataset/macdoc/val_dataset.json", attribute = 'length', tokenizer = None, mode = 'inference', size = -1, model_type = 'llama31'):
        self.dataset_path = dataset_path
        self.dataset = json.load(open(dataset_path,"r"))
        self.size = size 
        self.tokenizer = tokenizer
        self.attribute = attribute
        self.filter_by_attribute()
        self.mode = mode
        self.system_prompt = "You are an honest and to the point assistant, please follow the instruction and answer to the point"
        self.model_type = model_type
        if self.size != -1:
            self.dataset = dict(list(self.dataset.items())[:self.size])
            self.index_to_keys = list(self.dataset.keys())
        else:
            self.size = len(self.dataset)



    def alpaca_promp_format(self, src_text, instruction):
        return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n{instruction}\n\n{src_text}\n\nResponse:"
    def generate_attribute_specific_instruction(self,control_value):
        base_prompt = f"Write a summary of the source text."
        if self.attribute == 'length':
            ca_aspect = f"The summary should be {control_value} in length. The length is defined in terms of number of words used in the summary"
        elif self.attribute == 'extractiveness':
            ca_aspect = f"The summary should be {control_value} in extractiveness. Extractiveness is defined by the degree of exact copying from the source text"
        elif self.attribute == 'specificity':
            ca_aspect = f"The summary should be {control_value} in specificity. Specificity is defined by the degree of detail in the summary"
        elif self.attribute == 'topic':
            ca_aspect = f"The summary should be focussed on the topic {control_value}"
        elif self.attribute == 'Speaker':
            ca_aspect = f"The summary should be written from the perspective of {control_value}"
        #prompt = f"{base_prompt} {ca_aspect}. The source text is given below. "
        instruction = f"{base_prompt} {ca_aspect}. The source text is given below. "
        return instruction


    
    def filter_by_attribute(self):
        tmp_dataset = {}
        for key , value in self.dataset.items():
            if value['control_attribute'][self.attribute] != '':
                tmp_dataset[key] = value
        self.dataset = tmp_dataset
        self.index_to_keys = list(self.dataset.keys())


    def format_data_llama31(self, instruction, input_text, output):
        # Construct the full text in Llama 3.1 format
        full_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|> {self.system_prompt}<|eot_id|>"
        full_text += f"<|start_header_id|>user<|end_header_id|> {instruction}\n\n{input_text}<|eot_id|>"
        prompt = full_text + f"<|start_header_id|>assistant<|end_header_id|>"
        full_text += f"<|start_header_id|>assistant<|end_header_id|> {output}<|eot_id|>"

        # Find the start and end index of the output
        # output_start = full_text.rfind(output)
        # output_end = output_start + len(output)

        # Create labels
        #labels = [-1] * len(full_text)
        #labels[output_start:output_end] = list(output)  # Set labels to be exactly the same as output

        return full_text, prompt
    def format_data_mistral(self, instruction, input_text, output):
        #mistral doesn't require system prompt
        # Construct the full text in Mistral format
        full_text = f"<s>[INST] {instruction}"
        full_text += f" {input_text}"
        prompt = full_text + f" [/INST]"
        full_text += f" [/INST]{output}</s>"
        return full_text, prompt



    
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        src = self.dataset[self.index_to_keys[index]]['source']
        reference = self.dataset[self.index_to_keys[index]]['reference']
        attribute_value = self.dataset[self.index_to_keys[index]]['control_attribute'][self.attribute]
        instruction = self.generate_attribute_specific_instruction(attribute_value)
        if self.model_type == 'mistral':
            example , prompt = self.format_data_mistral(instruction, src, reference)
        elif self.model_type == 'llama31':
            example , prompt = self.format_data_llama31(instruction, src, reference)
        #example = prompt + reference

        tokenized_prompt = torch.tensor(
            self.tokenizer.encode(prompt, add_special_tokens = False), dtype=torch.int64
        )
        example = self.tokenizer.encode(example, add_special_tokens = False)
        #example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        #labels = torch.tensor(labels, dtype=torch.int64)

        labels = copy.deepcopy(example)
        #labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        #labels[~label_mask] = IGNORE_INDEX
        if self.mode == 'inference':
            return {
                "input_ids": tokenized_prompt.unsqueeze(0),
                #"labels": labels.unsqueeze(0),
                #"attention_mask":example_mask.unsqueeze(0),
                "output" : reference,
                "input": src,
                "prompt": prompt,
                "control_value": attribute_value,
                "control_attribute": self.attribute

            }
        else:

            return {
                "input_ids": example,
                "labels": labels,
                "attention_mask":example_mask
            }

def dpo_dataset(Dataset):

    def __init__(self, dataset_path = "/home2/tathagato/summarization/MACSUM/dataset/macdoc/train.json", attributes = ['length'], tokenizer = None, mode = 'train', size = -1):
        self.dataset_path = dataset_path
        self.data = json.load(open(dataset_path,"r"))
        self.size = size 
        self.tokenizer = tokenizer
        self.attributes = attributes
        self.generate_dpo_pairs()
        self.mode = mode

    def generate_dpo_pairs(self):
        dataset = {}
        new_idx = 0
        for idx, example in tqdm.tqdm(enumerate(data)):
            source = example['source']
            num_references = len(example['references'])
            for i in range(num_references):
                for j in range(i+1, num_references):
                    first_control_attributes = example['references'][i]['control_attribute']
                    second_control_attributes = example['references'][j]['control_attribute']
                    is_usable = True
                    for attr in attributes:
                        if first_control_attributes[attr] == "" or second_control_attributes[attr] == "":
                            is_usable = False
                            break
                        elif first_control_attributes[attr] == second_control_attributes[attr]:
                            is_usable = False
                            break
                    if is_usable:
                        first_selection = {attr : first_control_attributes[attr] for attr in attributes}
                        second_selection = {attr : second_control_attributes[attr] for attr in attributes}
                        dataset[new_idx] = { "source" : source , "chosen" : example['references'][i]['summary'], "unchosen" : example['references'][j]['summary'], "prefered_control_attributes" : first_selection, "rejected_control_attributes" : second_selection, "indexes" : [i, j]}
                        new_idx += 1
                        dataset[new_idx] = { "source" : source , "chosen" : example['references'][j]['summary'], "unchosen" : example['references'][i]['summary'], "prefered_control_attributes" : second_selection, "rejected_control_attributes" : first_selection, "indexes" : [j, i]}
                        new_idx += 1
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)

    def generate_attribute_specific_instruction(self,control_values):
        base_prompt = f"Write a summary of the source text."
        ca_aspects = []
        for attr in self.attributes:
            if attr == 'length':
                ca_aspect = f"The summary should be {control_values[attr]} in length. The length is defined in terms of number of words used in the summary"
            elif attr == 'extractiveness':
                ca_aspect = f"The summary should be {control_values[attr]} in extractiveness. Extractiveness is defined by the degree of exact copying from the source text"
            elif attr == 'specificity':
                ca_aspect = f"The summary should be {control_values[attr]} in specificity. Specificity is defined by the degree of detail in the summary"
            elif attr == 'topic':
                ca_aspect = f"The summary should be focussed on the topic {control_values[attr]}"
            elif attr == 'Speaker':
                ca_aspect = f"The summary should be written from the perspective of {control_values[attr]}"
            ca_aspects.append(ca_aspect)
        #convert all the strings in ca_aspects to a single string
        ca_aspects = ". ".join(ca_aspects)
        instruction = f"{base_prompt} {ca_aspects}. The source text is given below. "

        return instruction
    def alpaca_promp_format(self, src_text, instruction):
        return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n{instruction}\n\n{src_text}\n\nResponse:"

    
    def __getitem__(self, index):
        IGNORE_INDEX = -100
        source = self.dataset[index]['source']
        prefered_control_attributes = self.dataset[index]['prefered_control_attributes'] 
        instruction = self.generate_attribute_specific_instruction(prefered_control_attributes)
        prompt = self.alpaca_promp_format(source, instruction)
        prefered_response = self.dataset[index]['chosen']
        rejected_response = self.dataset[index]['unchosen']
        prefered_prompt = prompt + prefered_response
        rejected_prompt = prompt + rejected_response
        prefered_prompt = torch.tensor(
            self.tokenizer.encode(prefered_prompt), dtype=torch.int64
        )
        rejected_prompt = torch.tensor(
            self.tokenizer.encode(rejected_prompt), dtype=torch.int64
        )
        prefered_response = self.tokenizer.encode(prefered_response)




if __name__=='__main__':
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    #import huggingface tokenizers from transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #print the id and name of all then special tokens
    print(tokenizer.special_tokens_map)
    #print the id of all the special tokens
    print(tokenizer.all_special_ids)
    #print the name of all the special tokens
    print(tokenizer.all_special_tokens)
    
    dataset_path = "/home2/tathagato/summarization/MACSUM/dataset/macdoc/train_dataset.json"
    dataset = MACSUM(dataset_path, tokenizer= tokenizer, attribute = 'length', mode = "train")
    print(len(dataset))
    example = dataset[0]
    print(example.keys())
    #import code; code.interact(local=locals())
    input_ids = example['input_ids']
    print("text with special tokens")
    print(tokenizer.decode(input_ids))
    print("text without special tokens")
    print(tokenizer.decode(input_ids, skip_special_tokens=True))

    labels = example['labels']
    print(labels)
    print(input_ids)

    #get the sequence after the last -100 in labels
    new_labels = []
    for i in labels:
        if i != -100:
            new_labels.append(i)
    print(new_labels)
    print("text with special tokens")
    print(tokenizer.decode(new_labels))
    print("text without special tokens")
    print(tokenizer.decode(new_labels, skip_special_tokens=True))
