from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig,TrainerCallback
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
import torch
 
 
 
class AlpacaDataset(Dataset):
    def __init__(self, data):
        self.data = data
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item['instruction']
        input_text = item.get('input', '')
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        return prompt
 
def collate_fn(batch):
    return tokenizer(batch, padding=True, return_tensors="pt")
 
 
 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
 
tokenizer.pad_token = tokenizer.eos_token
 
 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
 
accelerator = Accelerator()
device = accelerator.device
# model.to(device)
 
data = [
    {"instruction": 'For a given input text, create a short length summary as per the following definition of length. Length indicates the number of words in the summary, the length of the summary can be one of the three categories of [short, normal, long]. Please generate a summary that captures the essential information and key points of the input text, within the chosen length constraint.', 
    "input": """
    (CNN)Mountaineers have returned to Mount Everest for this year's climbing season, resuming the quest to summit the world's highest peak after a deadly season last year. In 2014, the Nepal climbing season ended after a piece of glacial ice fell, unleashing an avalanche that killed 16 Nepalis who had just finished their morning prayers. The April 18 accident was the single deadliest incident to ever occur on Mount Everest. The deaths launched fierce debates about the enormous risks faced by the Sherpas and the dangers of climbing Everest. In order to reduce risks, the route through Khumbu Icefall, the notoriously treacherous path where the 16 were killed, has been changed to one that takes longer but is expected to be safer. "They're going in the icefall and, as we found out on April 18, it's the most dangerous place," said Conrad Anker, a veteran climber who has been to Everest three times. "They're exposed to the tumbling ice, hanging seracs above it. It's very, very dangerous. It's the most dangerous place I've been in the mountains." At this point in the season, climbing teams have not yet entered Khumbu Icefall, which is essentially a frozen river rapid with jagged pieces breaking off and moving. Nepal has issued 347 permits this year to climb Mount Everest, with 125 of them from the previously shortened season, according to the Nepal Ministry of Tourism. It's a slight increase from the 334 who were given permission last year. The local Nepalese committee that determines the path up Everest announced in February that a different route had been selected. The climbers will now take a central route through the Khumbu Icefall, avoiding the area where the deaths occurred. The committee comprised of Sherpas voted to return to the central route for safety reasons. "There will be little risk of avalanche than in the right or left," said Yangji Doma Sherpa, the spokeswoman for the Sagarmatha Pollution Control Committee. The central route had been used in the 1990s, but was abandoned in favor of a quicker route, she said. The new path means climbers will have to cross more crevasses, and use more vertical and horizontal ladders. The committee issued a recommendation that the weight of workers' gear be limited to avoid overloading the ladders. "I think it will be an hour longer on the icefall," said Alan Arnette, who is blogging from Everest base camp this season. "I don't think it will be game changer." But one company, Alpenglow Expeditions, said it would stop climbing from the Nepal side, where the climbers have to go through the icefall, in favor of the northern route from China. "We've seen it get progressively more dangerous over the last few years," said  Adrian Ballinger, the company's founder and CEO. "We believe the risk is too great for our workers." According to the China Tibet Mountaineering Association, 320 people have been registered to climb the northern route to Everest this year. That's 136 more than last year. The Chinese side of Everest has typically been less popular than its Nepal counterpart, because of concerns of government closures. Some Everest observers say the northern route has harsher weather and more rocky terrain, but it also doesn't have an icefall. The increasing popularity of the northern route has caused concern amongst Nepali companies that climbers will divert to the Chinese side. "I can already see the shift with mountaineers I speak to," said Dawa Steven Sherpa, who is based in Nepal. "More people are going to go to Tibet than Nepal. Nepal needs the tourism far more than China does. China has incredible wealth of resources and Nepal does not." Leading expeditions is how Sherpas feed their families and send their children to school. Nepal depends heavily on tourism dollars. Many of the guides had to bury their friends after the accident last year, and while they may be ready to return to the summit, their families are not. Many of them are "leaving behind nervous, stressed-out wives and children," whose memories of what happened last year are fresh, said Dawa Sherpa, managing director of Asian Trekking. "They do say they don't want to put them through that again," he said. "They're not fearful for their own lives, it's what they're putting their family through." Several mountaineers are also returning this year. One of them is Jon Reiter, who spoke to CNN last year after the tragedy. When the icy avalanche thundered down, Reiter was shoved behind an ice block by his Sherpa guide. Reiter, who is making his way to base camp this year, could not be reached directly. But he explained why he's heading back to Everest this year on his blog. "I can't quite find the words to tell you why, or what really pulls me back to the mountains," he wrote. "When we were in the midst of last year's events it was hard to see the big picture. It was hard to remember that people die in the mountains but that it's more rare than not. "It was hard for me to remember that I'm not choosing between my life at home and dying in the mountains. I like to think it's similar to surviving a plane crash or a major pile up on the freeway." CNN's Sugam Pokharel contributed to this report. """, 
    "output": "No idea" } # Not required for now
]
dataset = AlpacaDataset(data)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
item = next(iter(dataloader))
import code; code.interact(local=locals())
 
 
 
model.eval()
results = []
with torch.no_grad():
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model.generate(**batch, max_new_tokens=100)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded_outputs)
 
# Print results
for i, result in enumerate(results):
    print(f"Result {i+1}: {result}")
