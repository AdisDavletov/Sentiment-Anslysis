import re
import os
import codecs
import numpy as np
import scipy.sparse as csr
from collections import defaultdict
from scipy.special import expit


def tokenizer(list_of_strings):

    list_of_lists_of_tokens = list()

    with open("Starter code/stopwords.txt", "r") as f:
        STOPWORDS = map(lambda x: x.strip(), f.readlines())

    STOPWORDS = set(STOPWORDS)

    for idx in range(len(list_of_strings)):
        tmp_string = list_of_strings[idx].lower()
        tmp = re.findall('\w+', tmp_string)
        buff = []
        for x in tmp:
            if not x in STOPWORDS:
                buff.append(x)
        list_of_lists_of_tokens.append(buff)

    return list_of_lists_of_tokens

def sigmoid(x):
    return .5 * (1 + np.tanh(.5 * x)) #1.0 / (1.0 + np.exp(-x))

def accuracy(y_pred, y_true):
    correct = sum(y1 == y2 for y1, y2 in zip(y_pred, y_true))
    acc = 100.0 * correct / len(y_true)
    return acc

def fit_theta(x, y, theta, alpha, learning_rate, iters_num = 15000, cost_checking_time = 100, adagrad = True):

    N, V = np.shape(x)
    G = np.array([0.0 for x in range(V)])
    tmp = np.array([0 for x in range(V)])
    cost_prev = cost(x, y, theta, alpha)
    eps = 0.00000001

    percentage = 0
    reminder = 0
    
    for i in range(iters_num):
        gradient = grad(x, y, theta, alpha)
        
        if adagrad:
            G += gradient ** 2
            eta = learning_rate / np.sqrt(G + eps)
            theta = theta - eta * gradient
        else:
            theta = theta - learning_rate * gradient

        reminder += 1
        if reminder == cost_checking_time:
            cost_curr = cost(x, y, theta, alpha)
            percentage += 1
            
            if abs(cost_prev - cost_curr) < 0.0001:
                break
            
            if cost_curr < 0.01:
                break
            cost_prev = cost_curr
            reminder = 0

    print('Number of iterations:', i)
    return theta

def init_theta(size):
    return np.array([0 for x in range(size)])

def cost(x, y, theta, alpha):

    N = len(y)
    reg_contrib = np.sum(theta[1:] ** 2 * alpha)
    tmp = sigmoid(x.dot(theta))

    pos = np.log(tmp + 0.00000001)
    neg = np.log(1 - tmp + 0.00000001)
    ny = 1 - y

    L = (-1 / N) * (y.dot(pos) + ny.dot(neg)) + reg_contrib
    
    return L

def grad(x, y, theta, alpha):
    N = len(y)
    gradient = x.transpose().dot(sigmoid(x.dot(theta)) - y) / N

    tmp = theta * (alpha * 2)
    tmp[0] = 0

    gradient += tmp

    return gradient

def fit_params(x, y):
    N, V = x.shape
    rows = np.random.permutation(N)
    delimiter = int(np.ceil(0.8 * N))

    x_train = x[rows[:delimiter], :]
    x_dev = x[rows[delimiter:], :]
    y_train = y[rows[:delimiter]]
    y_dev = y[rows[delimiter:]]

    max_acc = 0.0
    alphas = [0.00002, 0.00003, 0.000009, 0.0001]
    learning_rates = [0.01, 0.003, 0.001]
    params = (0, 0)

    for alpha in alphas:
        for learning_rate in learning_rates:
            
            theta = init_theta(V)
            theta = fit_theta(x_train, y_train, theta, alpha, learning_rate)

            preds = list(map(lambda x: 1 if x >= 0 else 0, x_dev.dot(theta)))

            acc = accuracy(preds, y_dev)
            print(alpha, learning_rate,':', acc)
            if acc > max_acc:
                params = (alpha, learning_rate)
                max_acc = acc

    print('Results of fitting:', '(alpha, learning_rate) =', params)

    return params


def train(train_texts, train_labels, enable_params_fitting = False):
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param train_texts: a list of texts (str objects), one str per example
    :param train_labels: a list of labels, one label per example
    :param alpha: a float number for regularization
    :param 
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """

    res = {}

    texts = tokenizer(train_texts)

    # | Retrieving features from texts

    data = []
    indices = []
    indptr = [0]
    vocabulary = {}

    for text in texts:
        bias = vocabulary.setdefault('BIAS', len(vocabulary))
        indices.append(bias)
        data.append(1)

        # | unigrams
        for word in text:
            index = vocabulary.setdefault(word, len(vocabulary))
            indices.append(index)
            data.append(1)
        '''
        # | bigrams
        for i in range(len(text) - 1):
            index = vocabulary.setdefault(text[i] + ' ' + text[i + 1], len(vocabulary))
            indices.append(index)
            data.append(1)
        '''
        indptr.append(len(indices))

    x = csr.csr_matrix((data, indices, indptr), dtype = int)

    # | Retrieving labels
    y = np.array(list(map(lambda x: 1 if x == 'pos' else 0, train_labels)))

    if enable_params_fitting:
        alpha, learning_rate = fit_params(x, y)
    else:
        alpha, learning_rate = (0.00001, 0.001)

    # | Initialization and fitting of theta
    theta = init_theta(len(vocabulary))
    theta = fit_theta(x, y, theta, alpha, learning_rate, adagrad = False)

    res['theta'] = theta
    res['vocabulary'] = vocabulary
    res['shape'] = x.shape

    return res

def print_weigts(words_dict, theta, reverse = False, maxcount = 20):
    
    words = list(words_dict)
    vocabulary = defaultdict(int)

    for i in range(len(words_dict)):
        vocabulary[words[i]] = theta[i]

    if reverse:
        sorted_vocab = sorted(vocabulary, key = tmp.get, reverse = True)[:maxcount]
        sorted_theta = sorted(theta, reverse = True)[:maxcount]
    else:
        sorted_vocab = sorted(vocabulary, key = tmp.get)[:maxcount]
        sorted_theta = sorted(theta)[:maxcount]

    print(sorted_vocab)
    print(sorted_theta)


def classify(texts, params):
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """

    texts = tokenizer(texts)

    vocabulary = params['vocabulary']
    theta = params['theta']
    N, V = params['shape']

    data = []
    indices = []
    indptr = [0]

    for text in texts:
        bias = vocabulary.setdefault('BIAS', len(vocabulary))
        indices.append(bias)
        data.append(1)

        # | unigrams
        for word in text:
            index = vocabulary.setdefault(word, len(vocabulary))
            if index < V:
                indices.append(index)
                data.append(1)
        '''
        # | bigrams
        for i in range(len(text) - 1):
            index = vocabulary.setdefault(text[i] + ' ' + text[i + 1], len(vocabulary))
            if index < V:
                indices.append(index)
                data.append(1)
        '''
        indptr.append(len(indices))

    x = csr.csr_matrix((data, indices, indptr), dtype = int, shape = (len(texts), V))

    labels = list(map(lambda x: 'pos' if x >= 0 else 'neg', x.dot(theta)))

    return labels

