from os.path import join as pjoin
import numpy as np
import json
import tensorflow as tf
import logging

PAD_ID = 0


def get_indicies_sorted_by_context_length(data):
    context_lengths = np.array([len(context) for context in data["context"]])
    return np.argsort(context_lengths)


def reindex_dictionary(data, indices):
    for k, v in data.items():
        data[k] = v[indices]
    return data


def load_and_preprocess_data(data_dir):
    # Load the training data
    train = {}

    train_context = [list(map(int, line.strip().split()))
                     for line in open(pjoin(data_dir, "train.ids.context"))]

    train_question = [list(map(int, line.strip().split()))
                      for line in open(pjoin(data_dir, "train.ids.question"))]

    train_word_context = [line for line in open(pjoin(data_dir, "train.context"))]
    train_answer_span = [list(map(int, line.strip().split()))
                         for line in open(pjoin(data_dir, "train.span"))]
    train["context"] = np.array(train_context)
    train["question"] = np.array(train_question)
    train["word_context"] = np.array(train_word_context)
    train["answer_span_start"] = np.array(train_answer_span)[:, 0]
    train["answer_span_end"] = np.array(train_answer_span)[:, 1]

    train_indicies = get_indicies_sorted_by_context_length(train)
    train = reindex_dictionary(train, train_indicies)

    # Load the val data
    val = {}

    val_context = [list(map(int, line.strip().split()))
                   for line in open(pjoin(data_dir, "val.ids.context"))]
    val_question = [list(map(int, line.strip().split()))
                    for line in open(pjoin(data_dir, "val.ids.question"))]
    val_word_context = [line for line in open(pjoin(data_dir, "val.context"))]
    val_answer_span = [list(map(int, line.strip().split()))
                       for line in open(pjoin(data_dir, "val.span"))]
    val_answer = [line for line in open(pjoin(data_dir, "val.answer"))]

    val["context"] = np.array(val_context)
    val["question"] = np.array(val_question)
    val["word_context"] = np.array(val_word_context)
    val["answer_span_start"] = np.array(val_answer_span)[:, 0]
    val["answer_span_end"] = np.array(val_answer_span)[:, 1]
    val["word_answer"] = np.array(val_answer)

    val_indicies = get_indicies_sorted_by_context_length(val)
    val = reindex_dictionary(val, val_indicies)

    return train, val


def load_word_embeddings(data_dir):
    return np.load(pjoin(data_dir, "glove.trimmed.100.npz"))["glove"].astype(np.float32)
