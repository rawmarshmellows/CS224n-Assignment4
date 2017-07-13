# Evaluation script from SQuAD v1.1
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    if normalize_answer(prediction) == normalize_answer(ground_truth):
#         print(normalize_answer(prediction))
#         print(normalize_answer(ground_truth))
        return(True)
    else:
        return(False)

def evaluate(predictions, ground_truths):     
    f1 = exact_match = total = no_answer = 0
    for prediction, ground_truth in zip(predictions, ground_truths):
#         dprint("prediction is: {}".format(prediction), debug = True)
#         dprint("ground_truth is: {}".format(ground_truth), debug = True)
        total += 1
        
        # sometimes we might not predict the answer so in this case we just skip the loop iteration
        if prediction == "": 
            no_answer += 1
            continue
        
        # now we need to turn the lists into strings
        exact_match += exact_match_score(prediction, ground_truth)
        f1 += f1_score(prediction, ground_truth)
        
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {"f1" : f1,
            "EM" : exact_match,
            "no_answer" : no_answer}