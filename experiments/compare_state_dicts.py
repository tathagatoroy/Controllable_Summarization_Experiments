import torch 
import pprint
#a function which takes two state dicts as input and return the layers which contain different weights
def compare_state_dicts(dict1, dict2):
    diff_layers = []
    for key in dict1.keys():
        if torch.all(torch.eq(dict1[key], dict2[key])):
            continue
        else:
            diff_layers.append(key)
    return diff_layers

if __name__=="__main__":
    path_2 = "raw_model_after_first_training.pth"
    path_1 = "raw_model.pth"
    path_3 = "raw_model_after_second_training.pth"
    model_1 = torch.load(path_1)
    model_2 = torch.load(path_2)
    model_3 = torch.load(path_3)
    diff_layers = compare_state_dicts(model_1, model_2)
    pprint.pprint(diff_layers)
    diff_layers = compare_state_dicts(model_2, model_3)
    print("----------------------------------------------------------------------------------------------")
    pprint.pprint(diff_layers)
    
    