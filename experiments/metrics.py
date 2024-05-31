#evaluation metrics considered for our experiments
#note that code are borrowed from the following repository:
#   1. https://github.com/yxuansu/SimCTG/blob/main/simctg/evaluation.py
#   2. https://github.com/salesforce/hydra-sum/tree/main/postprocessing

import textstat
from nltk import word_tokenize, ngrams
import os
import pandas as pd
import csv
import numpy as np
import pickle as pkl
import multiprocessing as mp
import tqdm
import nltk
from evaluate import load
import shutil
import time
from multiprocessing import Pool
from pyrouge import Rouge155
import tempfile
import pwd

nltk.download('punkt')


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
        print("sumary_tokens is zero")
        print(article)
        print(summary)
        return 0

    coverage = float(sum([len(f) for f in frags])) / float(len(summary_tokens))
    return coverage

def get_summary_length(article, summary):
    return len(word_tokenize(summary.lower()))

def get_compression_ratio(article, summary):
    if len(word_tokenize(article.lower())) == 0:
        print("article_tokens is zero")
        print(article)
        print(summary)
        return 0
    return float(len(word_tokenize(summary.lower()))) / float(len(word_tokenize(article.lower())))




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
        print("fragment density sumary_tokens is zero")
        print(article)
        print(summary)
        return 0
    density = float(sum([len(f)**2 for f in frags])) / float(len(summary_tokens))
    return density

def get_overlap(inp, out, ngram):
    grams_inp = set(ngrams(word_tokenize(inp.lower()), ngram))
    grams_out = set(ngrams(word_tokenize(out.lower()), ngram))

    total = len(grams_out)
    common = len(grams_inp.intersection(grams_out))
    if total == 0:
        return 0
    else:
        return float(common) / float(total)

