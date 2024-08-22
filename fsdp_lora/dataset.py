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
    def __init__(self, dataset_path = "/home2/tathagato/summarization/MACSUM/dataset/macdoc/val_dataset.json", attribute = 'length', tokenizer = None, mode = 'inference', size = -1):
        self.dataset_path = dataset_path
        self.dataset = json.load(open(dataset_path,"r"))
        self.size = size 
        if self.size != -1:
            self.dataset = dict(list(self.dataset.items())[:self.size])
        else:
            self.size = len(self.dataset)
        self.tokenizer = tokenizer
        self.attribute = attribute
        self.filter_by_attribute()
        self.mode = mode




    def alpaca_promp_format(self, src_text, instruction):
        return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n{instruction}\n\n{src_text}\n\nResponse:"
    def generate_attribute_specific_instruction(self,control_value):
        base_prompt = f"Write a summary of the source text."
        if self.attribute == 'length':
            ca_aspect = f"The summary should be {control_value} in length. The length is defined in terms of number of words used in the summary"
        elif controllable_aspect == 'extractiveness':
            ca_aspect = f"The summary should be {control_description} in extractiveness. Extractiveness is defined by the degree of exact copying from the source text"
        elif controllable_aspect == 'specificity':
            ca_aspect = f"The summary should be {control_description} in specificity. Specificity is defined by the degree of detail in the summary"
        elif controllable_aspect == 'topic':
            ca_aspect = f"The summary should be focussed on the topic {control_description}"
        elif controllable_aspect == 'Speaker':
            ca_aspect = f"The summary should be written from the perspective of {control_description}"
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


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        src = self.dataset[self.index_to_keys[index]]['source']
        reference = self.dataset[self.index_to_keys[index]]['reference']
        attribute_value = self.dataset[self.index_to_keys[index]]['control_attribute'][self.attribute]
        instruction = self.generate_attribute_specific_instruction(attribute_value)
        prompt = self.alpaca_promp_format(src, instruction)
        example = prompt + reference

        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        if self.mode == 'inference':
            return {
                "input_ids": prompt.unsqueeze(0),
                #"labels": labels.unsqueeze(0),
                #"attention_mask":example_mask.unsqueeze(0),
                "reference" : reference
            }
        else:

            return {
                "input_ids": example,
                "labels": labels,
                "attention_mask":example_mask,
                "reference" : reference
            }

# if __name__=='__main__':
#     model_name = "meta-llama/Meta-Llama-3.1-8B"
#     #import huggingface tokenizers from transformers
#     from transformers import AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     dataset_path = "/home2/tathagato/summarization/MACSUM/dataset/macdoc/train_dataset.json"
#     dataset = MACSUM(dataset_path, tokenizer, attribute = 'length')
#     print(len(dataset))
#     example = dataset[0]
#     print(example.keys())
#     #import code; code.interact(local=locals())
#     input_ids = example['input_ids']
#     print("text with special tokens")
#     print(tokenizer.decode(input_ids))
#     print("text without special tokens")
#     print(tokenizer.decode(input_ids, skip_special_tokens=True))

#     labels = example['labels']
#     print(labels)
#     print(input_ids)

#     #get the sequence after the last -100 in labels
#     new_labels = []
#     for i in labels:
#         if i != -100:
#             new_labels.append(i)
#     print(new_labels)
#     print("text with special tokens")
#     print(tokenizer.decode(new_labels))
#     print("text without special tokens")
#     print(tokenizer.decode(new_labels, skip_special_tokens=True))
