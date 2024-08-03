import json
import os
import sys
import time
import tqdm
import nltk
nltk.download('punkt')
from nltk import word_tokenize, ngrams
import sys
from pyrouge import Rouge155
from multiprocessing import Pool
import nltk
import shutil
import time
import tempfile
import pwd
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from nltk.corpus import stopwords
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
st_words = stopwords.words('english')

#rouge evaluations
def get_rouge_score(cand, ref, keys, output_dir = None):
    cand = [line.strip() for line in cand]
    ref = [line.strip() for line in ref]

    assert len(cand) == len(ref) and len(cand) == len(keys)

    candidates_chunks = list(chunks(cand, 1))
    references_chunks = list(chunks(ref, 1))
    keys_chunks = list(chunks(keys, 1))

    n_pool = 36
    arg_lst = []
    # print("number of chunks: ", len(candidates_chunks))
    for i in range(len(candidates_chunks)):
        arg_lst.append((candidates_chunks[i], references_chunks[i], keys_chunks[i], i, output_dir))
    pool = Pool(n_pool)
    final_results = {}
    results = pool.map(process, tqdm.tqdm(arg_lst))
    for i, r in enumerate(results):
        final_results[keys_chunks[i][0]] = r
    pool.close()
    #cleanup_tmp_dir(tmp_dir)
    cleanup_tmp_dir(output_dir)
    return final_results


def get_average_rouge_scores(cand, ref, output_dir = None):
    #keys can be empty not stictly required 
    keys = [str(i) for i in range(len(cand))]
    results = get_rouge_score(cand, ref, keys, output_dir)
    rouge_1 = 0
    rouge_2 = 0
    rouge_l = 0
    for k in results:
        rouge_1 += results[k]['rouge_1_f_score']
        rouge_2 += results[k]['rouge_2_f_score']
        rouge_l += results[k]['rouge_l_f_score']
    rouge_1 = rouge_1 / len(results)
    rouge_2 = rouge_2 / len(results)
    rouge_l = rouge_l / len(results)
    return rouge_1, rouge_2, rouge_l



def get_current_process_owner():
    return pwd.getpwuid(os.getuid()).pw_name

def get_directory_owner(directory):
    return pwd.getpwuid(os.stat(directory).st_uid).pw_name

def cleanup_tmp_dir(tmp_dir):
    print("cleaning tmp")
    all_dirs = [os.path.join(tmp_dir,f) for f in os.listdir(tmp_dir) if f.startswith("tmp")]
    current_process_owner = get_current_process_owner()
    for d in tqdm.tqdm(all_dirs):
        directory_owner_name = get_directory_owner(d)
        if directory_owner_name == current_process_owner:
            shutil.rmtree(os.path.join("/tmp", d))
    
def process(data):
    temp_dir = tempfile.mkdtemp(dir = "/tmp")
    candidates, references, keys, pool_id, output_dir = data
    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = "rouge-tmp-{}-{}".format(current_time, pool_id)
    if output_dir is not None:

        tmp_dir = os.path.join(output_dir, tmp_dir)
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:
        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])

        r = Rouge155()
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate(rouge_args="-e /home2/tathagato/summarization/ROUGE/pyrouge/tools/ROUGE-1.5.5/data -c 95 "
                                                          "-m -n 3 -a")
        # print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
        final_res = {"rouge_1_f_score": results_dict["rouge_1_f_score"], "rouge_2_f_score": results_dict["rouge_2_f_score"],"rouge_l_f_score": results_dict["rouge_l_f_score"],"rouge_3_f_score" : results_dict["rouge_3_f_score"],
                     "rouge_1_recall": results_dict["rouge_1_recall"], "rouge_2_recall": results_dict["rouge_2_recall"],"rouge_l_recall": results_dict["rouge_l_recall"], "rouge_3_recall": results_dict["rouge_3_recall"],
                     "rouge_1_precision": results_dict["rouge_1_precision"], "rouge_2_precision": results_dict["rouge_2_precision"],"rouge_l_precision": results_dict["rouge_l_precision"], "rouge_3_precision": results_dict["rouge_3_precision"],
                    }
    except Exception as e:
        print(e)
        print("error in processing")
        final_res = {"rouge_1_f_score": 0, "rouge_2_f_score": 0, "rouge_l_f_score": 0,
                     "rouge_1_recall": 0, "rouge_2_recall": 0, "rouge_l_recall": 0,
                     "rouge_1_precision": 0, "rouge_2_precision": 0, "rouge_l_precision": 0
                    }
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
        shutil.rmtree(temp_dir)
        
    return final_res

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
def get_rouge(predictions, references):
    #keys can be empty not stictly required 
    results = {}
    print("computing rouge")
    rouge_1, rouge_2, rouge_l = get_average_rouge_scores(predictions, references)
    results['rouge_1'] = rouge_1
    results['rouge_2'] = rouge_2
    results['rouge_l'] = rouge_l
    print("done computing rouge")
    return results

def get_evaluation_attribute(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    attribute = data['0']['control_attribute']
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
        print("fragment density sumary_tokens is zero")
        print(article)
        print(summary)
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


def get_control_value_for_length(predictions, summaries):
    """
    CER for length as defined by https://arxiv.org/pdf/2211.05041

    Args:
        summaries (Lis): the list of gold_summaries
        predictions (List): the list of predicted summaries

    Returns:
        str: average of cer scores
    """
    length_cers = []
    eta = 0.1
    for prediction, summary in zip(predictions, summaries):
        prediction_tokens = len(word_tokenize(prediction.lower()))
        summary_tokens = len(word_tokenize(summary.lower()))
        if summary_tokens == 0:
            cer = abs(prediction_tokens - summary_tokens) / (summary_tokens + eta)
        else:
            cer = abs(prediction_tokens - summary_tokens) / summary_tokens
        length_cers.append(cer)
    return sum(length_cers) / len(length_cers)


    



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
    result = {}
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
        print("control_value", key, len(output_dict[key]['article']), len(output_dict[key]['summary']))
        print("Prediction Summary Length", sum(output_dict[key]['prediction_summary_length'])/ len(output_dict[key]['prediction_summary_length']))
        print("Reference Summary Length", sum(output_dict[key]['reference_summary_length'])/ len(output_dict[key]['reference_summary_length']))
        print("Prediction Compression Ratio", sum(output_dict[key]['prediction_compression_ratio'])/ len(output_dict[key]['prediction_compression_ratio']))
        print("Reference Compression Ratio", sum(output_dict[key]['reference_compression_ratio'])/ len(output_dict[key]['reference_compression_ratio']))
        result[key] = {
            "num_examples": len(output_dict[key]['article']),
            "prediction_summary_length": sum(output_dict[key]['prediction_summary_length'])/ len(output_dict[key]['prediction_summary_length']),
            "reference_summary_length": sum(output_dict[key]['reference_summary_length'])/ len(output_dict[key]['reference_summary_length']),
            "prediction_compression_ratio": sum(output_dict[key]['prediction_compression_ratio'])/ len(output_dict[key]['prediction_compression_ratio']),
            "reference_compression_ratio": sum(output_dict[key]['reference_compression_ratio'])/ len(output_dict[key]['reference_compression_ratio'])
        }
    
    print("------------------------------")
    return result 
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
    
    result = {}
    for key in tqdm.tqdm(output_dict):
        print(key, len(output_dict[key]['article']), len(output_dict[key]['summary']), len(output_dict[key]['reference']))
        output_dict[key]['prediction_density'] = [get_fragment_density(article, summary) for article, summary in zip(output_dict[key]['article'], output_dict[key]['summary'])]
        output_dict[key]['prediction_coverage'] = [get_extractive_coverage(article, summary) for article, summary in zip(output_dict[key]['article'], output_dict[key]['summary'])]
        output_dict[key]['prediction_overlap'] = [get_overlap(article, summary) for article, summary in zip(output_dict[key]['article'], output_dict[key]['summary'])]
        output_dict[key]['reference_density'] = [get_fragment_density(article, summary) for article, summary in zip(output_dict[key]['article'], output_dict[key]['reference'])]
        output_dict[key]['reference_coverage'] = [get_extractive_coverage(article, summary) for article, summary in zip(output_dict[key]['article'], output_dict[key]['reference'])]
        output_dict[key]['reference_overlap'] = [get_overlap(article, summary) for article, summary in zip(output_dict[key]['article'], output_dict[key]['reference'])]

        print("control_value", key, len(output_dict[key]['article']), len(output_dict[key]['summary']))
        print("Prediction Density", sum(output_dict[key]['prediction_density'])/ len(output_dict[key]['prediction_density']))
        print("Prediction Coverage", sum(output_dict[key]['prediction_coverage'])/ len(output_dict[key]['prediction_coverage']))
        print("Prediction Overlap", sum(output_dict[key]['prediction_overlap'])/ len(output_dict[key]['prediction_overlap']))


        print("Reference Density", sum(output_dict[key]['reference_density'])/ len(output_dict[key]['reference_density']))
        print("Reference Coverage", sum(output_dict[key]['reference_coverage'])/ len(output_dict[key]['reference_coverage']))
        print("Reference Overlap", sum(output_dict[key]['reference_overlap'])/ len(output_dict[key]['reference_overlap']))
        result[key] = {
            "num_examples": len(output_dict[key]['article']),
            "prediction_density": sum(output_dict[key]['prediction_density'])/ len(output_dict[key]['prediction_density']),
            "prediction_coverage": sum(output_dict[key]['prediction_coverage'])/ len(output_dict[key]['prediction_coverage']),
            "prediction_overlap": sum(output_dict[key]['prediction_overlap'])/ len(output_dict[key]['prediction_overlap']),
            "reference_density": sum(output_dict[key]['reference_density'])/ len(output_dict[key]['reference_density']),
            "reference_coverage": sum(output_dict[key]['reference_coverage'])/ len(output_dict[key]['reference_coverage']),
            "reference_overlap": sum(output_dict[key]['reference_overlap'])/ len(output_dict[key]['reference_overlap'])
        }
    return result




def output_length_metrics(data):
    keys = list(data.keys())
    for key in tqdm.tqdm(keys):
        if 'generated_text' not in data[key]:
            #remove the element from dict
            print("popping key", key)
            data.pop(key)
            continue
        summary = data[key]['generated_text'].split("\n")[-1]
        data[key]['predicted_summary'] = summary

    result = get_length_stats(data)
    segregrated_data = {}
    keys = list(data.keys())
    summaries = []
    references = []
    for key in keys:
        control_value = data[key]['control_value']
        if control_value not in segregrated_data:
            segregrated_data[control_value] = {"summaries" : [], "articles" : [], "references" : []}
        segregrated_data[control_value]['summaries'].append(data[key]['predicted_summary'])
        segregrated_data[control_value]['articles'].append(data[key]['input'])
        segregrated_data[control_value]['references'].append(data[key]['output'])
        summaries.append(data[key]['predicted_summary'])
        references.append(data[key]['output'])

    for control_value in segregrated_data:
        print("control value ", control_value)
        cer = get_control_value_for_length(segregrated_data[control_value]['summaries'], segregrated_data[control_value]['references'])
        print("CER", cer)
        result[control_value]['cer'] = cer

    print("Overall")
    cer = get_control_value_for_length(summaries, references)
    print("CER", cer)
    result['overall_cer'] = cer
    

    # elif attribute == "extractiveness":
    #     data = clean_and_process_data(file)
    #     get_abstractive_data(data)
        
    
def get_summaries_articles_and_references(data):
    keys = list(data.keys())
    summaries = []
    articles = []
    references = []
    keys = list(data.keys())
    for key in tqdm.tqdm(keys):
        summaries.append(data[key]['generated_text'].split("\n")[-1])
        articles.append(data[key]['input'])
        references.append(data[key]['output'])
    return summaries, articles, references, keys

def get_control_error_for_extractiveness(data):
    """ computes the f_r for r extractiveness as defined  as defined by the https://arxiv.org/pdf/2211.05041 """
    summaries, articles, references, keys = get_summaries_articles_and_references(data)
    output_tmp_dir = "/tmp/rouge_output"
    if not os.path.isdir(output_tmp_dir):
        os.mkdir(output_tmp_dir)
    generated_rouge = get_rouge_score(summaries, articles, keys, output_tmp_dir)
    reference_rouge = get_rouge_score(references, articles, keys, output_tmp_dir)

    control_errors = []
    generated_frs = []
    reference_frs = []
    for index,key in enumerate(keys):
        generated_fr = (generated_rouge[key]['rouge_2_precision'] + generated_rouge[key]['rouge_3_precision']) / 2
        reference_fr = (reference_rouge[key]['rouge_2_precision'] + reference_rouge[key]['rouge_3_precision']) / 2
        if reference_fr == 0:
            control_error = abs(generated_fr - reference_fr) / (reference_fr + 0.1)
        else:
            control_error = abs(generated_fr - reference_fr) / reference_fr
            
        control_errors.append(control_error)
        generated_frs.append(generated_fr)
        reference_frs.append(reference_fr)
    return sum(control_errors) / len(control_errors), control_errors, generated_frs, reference_frs

def output_extractiveness_metrics(data):
    segregrated_data = {}
    keys = list(data.keys())
    result = get_abstractive_data(data)
    for key in keys:
        control_value = data[key]['control_value']
        if control_value not in segregrated_data:
            segregrated_data[control_value] = {}
        segregrated_data[control_value][key] = data[key]
    for control_value in segregrated_data:
        print("control value ", control_value)
        control_error, control_errors, generated_frs, reference_frs = get_control_error_for_extractiveness(segregrated_data[control_value])
        print("Control Error", control_error)
        print("num examples : ", len(control_errors))
        print("prediction F score : ",sum(generated_frs) / len(generated_frs))
        print("gold F score : ",sum(reference_frs) / len(reference_frs))
        result[control_value]['cer'] = control_error
        result[control_value]['prediction_f_score'] = sum(generated_frs) / len(generated_frs)
        result[control_value]['gold_f_score'] = sum(reference_frs) / len(reference_frs)

    overall_cer, control_errors, generated_frs, reference_frs = get_control_error_for_extractiveness(data)
    print("Overall Control Error", overall_cer)
    print("num examples : ", len(control_errors))
    print("prediction F score : ",sum(generated_frs) / len(generated_frs))
    print("gold F score : ",sum(reference_frs) / len(reference_frs))
    result['overall_cer'] = overall_cer
    result['overall_prediction_f_score'] = sum(generated_frs) / len(generated_frs)
    result['overall_gold_f_score'] = sum(reference_frs) / len(reference_frs)
    return result

    #get_abstractive_data(data)


#topic evaluation
def get_topic_values(topic, prediction):
    return [get_topic_value(x, y) for x, y in zip(topic, prediction)]

def get_topic_value(topic, prediction):
    topic_scores = []
    tokens = nltk.word_tokenize(topic)
    cnt_all = 0
    cnt_hit = 0
    for token in tokens:
        if not token.isalpha():
            continue
        cnt_all += 1
        if token.lower() in prediction.lower():
            cnt_hit += 1

    if cnt_all == 0:
        return 0

    topic_scores.append(1.0 * cnt_hit / cnt_all)
    return sum(topic_scores)/len(topic_scores)


def get_topic_score(data):
    topic_scores = []
    gold_scores = []
    for key in data:
        gold_summary = data[key]['output']
        prediction = data[key]['generated_text'].split("\n")[-1]
        topic_value = data[key]['control_value']
        topic_scores.append(get_topic_value(topic_value, prediction))
        gold_scores.append(get_topic_value(topic_value, gold_summary))
    abs_score = sum(topic_scores) / len(topic_scores)
    abs_gold_scores = sum(gold_scores) / len(gold_scores)
    relative_score = cal_diff(gold_scores, topic_scores)
    return relative_score, abs_score, abs_gold_scores

def cal_diff(gold_list, pred_list, relative=True):
    diffs = []
    for gold, pred in zip(gold_list, pred_list):
        diff = math.fabs(gold - pred)
        if relative:
            diff /= gold if gold else 0.1
        diffs.append(diff)
    ret = sum(diffs) / len(diffs)
    return ret

def output_topic_metrics(data):
    relative_score, abs_score, abs_gold_score = get_topic_score(data)
    print("Relative Score: ", relative_score)
    print("Absolute Prediction Score: ", abs_score)
    print("Absolute Gold Score: ", abs_gold_score)
    result = {}
    result['relative_score'] = relative_score
    result['abs_score'] = abs_score
    result['abs_gold_score'] = abs_gold_score
    return result

#specificity evaluation
def get_specificity_value(target):

    num_sent = len(nltk.sent_tokenize(target))
    target = nltk.word_tokenize(target.lower())
    target = [x for x in target if x not in st_words]
    target_pos = nltk.pos_tag(target)
    tot = len(target_pos)
    nn_words = [x for x, y in target_pos if y == 'NN']
    vb_words = [x for x, y in target_pos if y == 'VB']
    vbg_words = [x for x, y in target_pos if y == 'VBG']
    cd_words = [x for x, y in target_pos if y == 'CD']
    nn = len(nn_words)
    vb = len(vb_words)
    cd = len(cd_words)
    vbg = len(vbg_words)
    if num_sent == 0:
        return 0
    metrics = (0.1 * vbg + 0.2 * tot + 0.3 * nn + 0.4 * cd) / num_sent
    return metrics
def cal_intra(bucket_gold, bucket_pred, relative=True):
    diffs = []
    for prediction_specificity, gold_specificity in zip(bucket_pred, bucket_gold):
        diff = math.fabs(gold_specificity - prediction_specificity)
        if relative:
            diff /= gold_specificity
        diffs.append(diff)
    ret = sum(diffs) / len(diffs)
    return ret


def get_spe(data):
    bucket_len = []
    bucket_gold = []
    for key in tqdm.tqdm(data):
        gold_summary = data[key]['output']
        if 'generated_text' not in data[key]:
            continue
        prediction = data[key]['generated_text'].split("\n")[-1]
        if len(prediction) == 0:
            continue

        bucket_len.append(get_specificity_value(prediction))
        bucket_gold.append(get_specificity_value(gold_summary))
        

    intra_score = cal_intra(bucket_gold, bucket_len)
    f_pred = sum(bucket_len) / len(bucket_len)
    f_gold = sum(bucket_gold) / len(bucket_gold)
    return intra_score, f_pred, f_gold

def output_specificity_metrics(data):
    segregrated_data = {}
    result = {}
    for key in data:
        control_value = data[key]['control_value']
        if control_value not in segregrated_data:
            segregrated_data[control_value] = {}
        segregrated_data[control_value][key] = data[key]
    print(segregrated_data.keys())
    for key in segregrated_data:
        print("control key ", key)
        intra_score, f_pred, f_gold = get_spe(segregrated_data[key])
        print("CER : ", intra_score)
        print(f"Prediction Specificity : {f_pred}")
        print(f"Gold Specificity : {f_gold}")
        print("\n--")
        result[key] = {
            "cer" : intra_score,
            "prediction_specificity" : f_pred,
            "gold_specificity" : f_gold
        }
    intra_score, f_pred, f_gold = get_spe(data)
    print("overall CER : ", intra_score)
    print(f"Overall Prediction Specificity : {f_pred}")
    print(f"Overall Gold Specificity : {f_gold}")
    print("\n\n-----------------------------")
    result['overall_cer'] = intra_score
    result['overall_prediction_specificity'] = f_pred  
    result['overall_gold_specificity'] = f_gold
    return result


if __name__ == "__main__":
    directory = "/scratch/tathagato/non_packed_experiment_outputs"
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json")]
    attributes = [get_evaluation_attribute(file) for file in files]
    #sort file and attribute by attribute
    files = [x for _, x in sorted(zip(attributes, files))]
    print(f"num of files is {len(files)}")
    all_result = {}

    for file in tqdm.tqdm(files):
        attribute = get_evaluation_attribute(file)
        clean_and_process_data(file)
        print(f"Processing {file} and Attribute {attribute}")

        data = json.load(open(file, "r"))
        #take only the first 10 examples
        #data = {k: data[k] for k in list(data.keys())[:10]}

        if attribute == "length":
            result = output_length_metrics(data)
            all_result[file] = {'result' : result, 'attribute' : attribute}

            print("\n\n ----------------------length ----------------------\n\n")
        elif attribute == "extractiveness":
            result = output_extractiveness_metrics(data)
            all_result[file] = {'result' : result, 'attribute' : attribute}


            print("\n\n ----------------------extractiveness ----------------------\n\n")
        elif attribute == "topic":
            result = output_topic_metrics(data)
            all_result[file] = {'result' : result, 'attribute' : attribute}


            print("\n\n ----------------------topic ----------------------\n\n")
        elif attribute == "specificity":
            result = output_specificity_metrics(data)
            all_result[file] = {'result' : result, 'attribute' : attribute}

            print("\n\n ----------------------specificity ----------------------\n\n")
        else:
            print("Attribute not found")
            continue
        print(f"{file} done")
    output_file_name = "non_packed_combined_result.json"
    with open(output_file_name, "w") as f:
        json.dump(all_result, f)
    

    
    


        
    



