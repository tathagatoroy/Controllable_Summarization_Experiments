from transformers import pipeline

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
chatbot = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3", device = "cuda")
res = chatbot(messages, max_new_tokens = 200)
print(res)
import code; code.interact(local=locals())
