import os
import re
import numpy as np
from collections import defaultdict

def tokenizer(string, to_lower = True):
    stopwords = ['i', 'br', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'I', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
    if to_lower:
        string = string.lower()
    tokens = re.findall('\w+', string)
    res = []
    for token in tokens:
        if token not in stopwords:
            res.append(token)
    return res

def accuracy(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    correct = sum(y1 == y2 for y1, y2 in zip(y_pred, y_true))
    acc = 100 * correct / len(y_true)
    return acc

def make_vocabulary(texts):
    vocabulary = defaultdict(int)
    for text in texts:
        for word in text:
            vocabulary[word] += 1
    return vocabulary

def add_ngrams(tokens, n):
    buff = []
    for i in range(len(tokens) - n + 1):
        ngram = ''
        for j in range(n):
            ngram += (tokens[i + j] + ' ')
        buff.append(ngram.strip())
    return buff

def get_indices(labels, label):
    indices = []
    for i in range(len(labels)):
        if labels[i] == label:
            indices.append(i)
    return indices

def calculate_probabilities(texts, words_map):
    number_of_texts = len(texts)
    vocab_size = len(words_map)
    words_probs = np.zeros(vocab_size)
    for text in texts:
        tmp = np.zeros(vocab_size)
        for word in text:
            try:
                idx = words_map[word]
                tmp[idx] = 1
            except KeyError:
                continue
        words_probs += tmp
    words_probs += 1
    words_probs /= (number_of_texts + 2)    
    return words_probs

def errors(preds, labels, texts):
    texts_with_errors = []
    for i in range(len(labels)):
        if preds[i] != labels[i]:
            texts_with_errors.append(texts[i])
    return texts_with_errors

def init_texts(texts):
    texts = [tokenizer(string) for string in texts]
    texts = [unigrams + add_ngrams(unigrams, 2) for unigrams in texts]
    return texts

def make_vocabulary(texts):
    vocab = defaultdict(int)
    for text in texts:
        for word in text:
            vocab[word] += 1
    return vocab

def make_words_map(unique_words):
    words_map = {}
    for i in range(len(unique_words)):
        words_map[unique_words[i]] = i
    return words_map

def train(texts, labels, primary_vocab_size = 60000, secondary_vocab_size = 10000):
    texts = init_texts(texts)
    big_vocab = make_vocabulary(texts)
    unique_words = np.array(sorted(big_vocab, key = big_vocab.get, reverse = True)[:primary_vocab_size] + ['UNK'])    
    words_map = make_words_map(unique_words)

    pos_indices = get_indices(labels, 'pos')
    neg_indices = get_indices(labels, 'neg')
    pos_texts = [texts[i] for i in pos_indices]
    neg_texts = [texts[i] for i in neg_indices]

    pos_probs = calculate_probabilities(pos_texts, words_map)
    neg_probs = calculate_probabilities(neg_texts, words_map)
    posclass_prob = len(pos_texts) / len(texts)
    negclass_prob = len(neg_texts) / len(texts)
    
    bayes_weights = pos_probs / neg_probs
    sorted_indices = bayes_weights.argsort()
    
    new_unique_words = np.array(list(unique_words[sorted_indices[:secondary_vocab_size]]) + list(unique_words[sorted_indices[(len(sorted_indices) - secondary_vocab_size):]]) + ['UNK'])
    words_map = make_words_map(new_unique_words)

    pos_probs = calculate_probabilities(pos_texts, words_map)
    neg_probs = calculate_probabilities(neg_texts, words_map)
    
    params = {}
    params['pos'] = pos_probs
    params['neg'] = neg_probs
    params['posclass_prob'] = posclass_prob
    params['negclass_prob'] = negclass_prob
    params['words_map'] = words_map
    return params

def classify(texts, params):
    texts = init_texts(texts)
    pos_probs = params['pos']
    neg_probs = params['neg']
    posclass_prob = params['posclass_prob']
    negclass_prob = params['negclass_prob']
    words_map = params['words_map']
    vocab_size = len(words_map)
    
    npos_probs = 1 - pos_probs
    nneg_probs = 1 - neg_probs
    labels = []

    for text in texts:
        b = np.zeros(vocab_size)
        for word in text:
            try:
                b[words_map[word]] = 1
            except KeyError:
                b[vocab_size - 1] = 1
                continue
        nb = 1 - b

        pos_res = sum(np.log(pos_probs * b + npos_probs * nb)) + np.log(posclass_prob)
        neg_res = sum(np.log(neg_probs * b + nneg_probs * nb)) + np.log(negclass_prob)

        if np.exp(pos_res) > np.exp(neg_res):
            labels.append('pos')
        else:
            labels.append('neg')
    return labels
