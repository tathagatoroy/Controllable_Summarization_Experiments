import json
import os
import sys
import time
import tqdm
import nltk
nltk.download('punkt')
from nltk import word_tokenize, ngrams


def get_evaluation_attribute(filepath):
    basename = os.path.basename(filepath)
    filename = os.path.splitext(basename)[0]
    attribute = filename.split("_")[-1]
    return attribute
def get_fragment_density(article, summary):
    """
    Calculates the fragment density of a summary on an article.

    Density is defined as the average squared length of extracted fragments.

    Args:
        article (str): The article text.
        summary (str): The summary text.

    Returns:
        float: The fragment density of the summary on the article.
    """

    frags, article_tokens, summary_tokens = get_extractive_fragments(article, summary)
    if len(summary_tokens) == 0:
        # print("fragment density sumary_tokens is zero")
        # print(article)
        # print(summary)
        return 0
    density = float(sum([len(f)**2 for f in frags])) / float(len(summary_tokens))
    return density
def get_overlap(inp, out, ngram = 2):
    grams_inp = set(ngrams(word_tokenize(inp.lower()), ngram))
    grams_out = set(ngrams(word_tokenize(out.lower()), ngram))

    total = len(grams_out)
    common = len(grams_inp.intersection(grams_out))
    if total == 0:
        return 0
    else:
        return float(common) / float(total)
def get_extractive_fragments(article, summary):
    """
    Extracts fragments from an article that match sequences of words in a summary.

    Args:
        article (str): The article text.
        summary (str): The summary text.

    Returns:
        list: A list of lists, where each sublist represents a sequence of word indexes
            in the article that match a sequence in the summary.
        list: The tokenized article.
        list: The tokenized summary.
    """

    article_tokens = word_tokenize(article.lower())
    summary_tokens = word_tokenize(summary.lower())

    F = []  # List to store the extracted fragments
    i, j = 0, 0  # Indexes for iterating over article and summary tokens, respectively

    while i < len(summary_tokens):
        f = []  # List to store the current fragment
        while j < len(article_tokens):
            if summary_tokens[i] == article_tokens[j]:
                i_, j_ = i, j  # Store starting indexes of potential fragment
                #print(len(summary_tokens), len(article_tokens), i, j, i_, j_, summary_tokens[i_], article_tokens[j_])
                while (i_ < len(summary_tokens) and j_ < len(article_tokens)) and summary_tokens[i_] == article_tokens[j_]:
                    i_, j_ = i_ + 1, j_ + 1  # Update indexes while words match
                if len(f) < (i_ - i):  # Update fragment if a longer match is found
                    f = list(range(i, i_))
                j = j_  # Set j to the next position after the matched sequence
            else:
                j += 1  # Move to the next article token if no match found
        i += max(len(f), 1)  # Update i by the length of the extracted fragment or 1
        j = 1  # Reset j for the next iteration

        F.append(f)  # Append the extracted fragment to the list

    return F, article_tokens, summary_tokens



def get_extractive_coverage(article, summary):
    """
    Calculates the extractive coverage of a summary on an article.

    Coverage is defined as the ratio of words in the summary covered by fragments
    extracted from the article.

    Args:
        article (str): The article text.
        summary (str): The summary text.

    Returns:
        float: The extractive coverage of the summary on the article.
    """

    frags, article_tokens, summary_tokens = get_extractive_fragments(article, summary)
    if len(summary_tokens) == 0:
        #print("sumary_tokens is zero")
        return 0

    coverage = float(sum([len(f) for f in frags])) / float(len(summary_tokens))
    return coverage

def clean_and_process_data(file):
    data = json.load(open(file,"r"))
    keys = list(data.keys())
    for key in tqdm.tqdm(keys):
        #print(data[key].keys())
        if 'generated_text' not in data[key]:
            #remove the element from dict
            print("popping key", key)
            data.pop(key)
            continue
        summary = data[key]['generated_text'].split("\n")[-1]
        data[key]['predicted_summary'] = summary
    with open(file, "w") as f:
        json.dump(data, f)
    return data

def get_length_stats(data):
    output_dict = {}
    controlled_attribute = "length"
    keys = list(data.keys())
    for key in keys:
        control_value = data[key]['control_value']
        article = data[key]['input']
        summary = data[key]['predicted_summary']
        reference = data[key]['output']
        if control_value not in output_dict:
            output_dict[control_value] = {"article":[], "summary":[], "reference":[]}
        output_dict[control_value]['article'].append(article)
        output_dict[control_value]['summary'].append(summary)
        output_dict[control_value]['reference'].append(reference)
    for key in output_dict:
        output_dict[key]['prediction_summary_length'] = [get_summary_length(summary) for summary in output_dict[key]['summary']]
        output_dict[key]['prediction_compression_ratio'] = [get_compression_ratio(article, summary) for article, summary in zip(output_dict[key]['article'], output_dict[key]['summary'])]
        output_dict[key]['reference_summary_length'] = [get_summary_length(summary) for summary in output_dict[key]['reference']]
        output_dict[key]['reference_compression_ratio'] = [get_compression_ratio(article, summary) for article, summary in zip(output_dict[key]['article'], output_dict[key]['reference'])]
        print("control_value", key)
        print("Prediction Summary Length", sum(output_dict[key]['prediction_summary_length'])/ len(output_dict[key]['prediction_summary_length']))
        print("Reference Summary Length", sum(output_dict[key]['reference_summary_length'])/ len(output_dict[key]['reference_summary_length']))
        print("Prediction Compression Ratio", sum(output_dict[key]['prediction_compression_ratio'])/ len(output_dict[key]['prediction_compression_ratio']))
        print("Reference Compression Ratio", sum(output_dict[key]['reference_compression_ratio'])/ len(output_dict[key]['reference_compression_ratio']))
    
    print("------------------------------")
    #print(output_dict.keys())

def get_summary_length(summary):
    return len(word_tokenize(summary.lower()))

def get_compression_ratio(article, summary):
    if len(word_tokenize(article.lower())) == 0:
        print("article_tokens is zero")
        print(article)
        print(summary)
        return 0
    return float(len(word_tokenize(summary.lower()))) / float(len(word_tokenize(article.lower())))

def get_abstractive_data(data):
    keys = list(data.keys())
    print(data[keys[0]].keys())
    output_dict = {}
    controlled_attribute = "extractiveness"
    for key in tqdm.tqdm(keys):
        control_value = data[key]['control_value']
        article = data[key]['input']
        summary = data[key]['predicted_summary']
        reference = data[key]['output']
        if control_value not in output_dict:
            output_dict[control_value] = {"article":[], "summary":[], "reference":[]}
        output_dict[control_value]['article'].append(article)
        output_dict[control_value]['summary'].append(summary)
        output_dict[control_value]['reference'].append(reference)
    #print(output_dict.keys())
    #normal, high, fully
    #sort the key in reverse order
    output_dict = dict(sorted(output_dict.items(), reverse=True))
    for key in tqdm.tqdm(output_dict):
        print(key)
        output_dict[key]['prediction_density'] = [get_fragment_density(article, summary) for article, summary in zip(output_dict[key]['article'], output_dict[key]['summary'])]
        output_dict[key]['prediction_coverage'] = [get_extractive_coverage(article, summary) for article, summary in zip(output_dict[key]['article'], output_dict[key]['summary'])]
        output_dict[key]['prediction_overlap'] = [get_overlap(article, summary) for article, summary in zip(output_dict[key]['article'], output_dict[key]['summary'])]
        output_dict[key]['reference_density'] = [get_fragment_density(article, summary) for article, summary in zip(output_dict[key]['article'], output_dict[key]['reference'])]
        output_dict[key]['reference_coverage'] = [get_extractive_coverage(article, summary) for article, summary in zip(output_dict[key]['article'], output_dict[key]['reference'])]
        output_dict[key]['reference_overlap'] = [get_overlap(article, summary) for article, summary in zip(output_dict[key]['article'], output_dict[key]['reference'])]
        print("control_value", key)
        print("Prediction Density", sum(output_dict[key]['prediction_density'])/ len(output_dict[key]['prediction_density']))
        print("Prediction Coverage", sum(output_dict[key]['prediction_coverage'])/ len(output_dict[key]['prediction_coverage']))
        print("Prediction Overlap", sum(output_dict[key]['prediction_overlap'])/ len(output_dict[key]['prediction_overlap']))


        print("Reference Density", sum(output_dict[key]['reference_density'])/ len(output_dict[key]['reference_density']))
        print("Reference Coverage", sum(output_dict[key]['reference_coverage'])/ len(output_dict[key]['reference_coverage']))
        print("Reference Overlap", sum(output_dict[key]['reference_overlap'])/ len(output_dict[key]['reference_overlap']))




def output_metrics(file, attribute):
    if attribute == "length":
        data = clean_and_process_data(file)
        get_length_stats(data)
    elif attribute == "extractiveness":
        data = clean_and_process_data(file)
        get_abstractive_data(data)
        
    

