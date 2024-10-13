import os 
from openai import OpenAI
import json


def convert_chat_completion_to_json(chat_completion):
    # Convert the nested structure to a dictionary
    json_output = {
        "id": chat_completion.id,
        "model": chat_completion.model,
        "object": chat_completion.object,
        "created": chat_completion.created,
        "service_tier": chat_completion.service_tier,
        "system_fingerprint": chat_completion.system_fingerprint,
        "usage": {
            "completion_tokens": chat_completion.usage.completion_tokens,
            "prompt_tokens": chat_completion.usage.prompt_tokens,
            "total_tokens": chat_completion.usage.total_tokens,
            "prompt_tokens_details": chat_completion.usage.prompt_tokens_details,

        },
        "choices": [
            {
                "index": choice.index,
                "finish_reason": choice.finish_reason,
                "logprobs": choice.logprobs,
                "message": {
                    "role": choice.message.role,
                    "content": choice.message.content,
                    "function_call": choice.message.function_call,
                    "tool_calls": choice.message.tool_calls,
                    "refusal": choice.message.refusal
                }
            }
            for choice in chat_completion.choices
        ]
    }
    
     # Convert the dictionary to a JSON string
    #json_output = json.dumps(chat_completion_dict, indent=4)
    
    return json_output

def extract_answer(json_output):
    responses = [res['message']['content'] for res in json_output['choices']]
    return responses

if __name__ == "__main__":


    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
            ],
        max_tokens=20,
        temperature=0,
        top_p=1,
        )
    #save completion to a file
    save_dir = "openai_outputs"
    os.makedirs(save_dir, exist_ok=True)
    print(completion)
    json_output = convert_chat_completion_to_json(completion)
    import code; code.interact(local=locals())
    print(json_output)
    print(extract_answer(json_output))
    with open(f"{save_dir}/completion.json", "w") as f:
        json.dump(json_output, f, indent=4)
