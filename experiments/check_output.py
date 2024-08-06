
import json
import json
import pprint
def transform_data(data):
    for key in data:
        for sub_key in data[key]:
            data[key][sub_key]['generated_text'] = data[key][sub_key]['summary']
    #print(data['first attribute']['0']['generated_text'], "\n", data['first attribute']['0']['summary'])
    return data
directory = "/scratch/tathagato/test_cascaded_lora_outputs/"
import os
files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(".json")]
print(len(files))
datas = []
import json
for file in files:
    with open(file, "r") as f:
        data = json.load(f)
        data = transform_data(data)
        datas.append(data)

from eval import output_length_metrics, output_extractiveness_metrics
#output file for printf
for index,data in enumerate(datas):
    print("processing file", files[index])
    first_attribute = data['first attribute']
    second_attribute = data['second attribute']
    print("length metrics for first attribute")
    output_length_metrics(first_attribute)
    print("extractiveness metrics for first attribute")
    output_extractiveness_metrics(first_attribute)
    print("length metrics for second attribute")
    output_length_metrics(second_attribute)
    print("extractiveness metrics for second attribute")
    output_extractiveness_metrics(second_attribute)
    print("\n---------------------------------------------------------\n")

