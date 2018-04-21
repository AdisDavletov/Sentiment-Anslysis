import re
import numpy as np
import scipy.sparse as csr
from collections import defaultdict

def init_texts(texts):
    texts = [tokenizer(text) for text in texts]
    texts = [unigrams + add_ngrams(unigrams, 2) for unigrams in texts]
    return texts

def add_ngrams(tokens, n):
    buff = []
    for i in range(len(tokens) - n + 1):
        ngram = ''
        for j in range(n):
            ngram += (tokens[i + j] + ' ')
        buff.append(ngram.strip())
    return buff

def tokenizer(text, to_lower = True):
    with open("Starter code/stopwords.txt", "r") as f:
        stopwords = map(lambda x: x.strip(), f.readlines())
    buff = []
    if to_lower:
        text = text.lower()
    text = re.findall('\w+', text)
    for word in text:
        if not word in stopwords:
            buff.append(word)
    return buff

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def accuracy(y_pred, y_true):
    correct = sum(y1 == y2 for y1, y2 in zip(y_pred, y_true))
    acc = 100.0 * correct / len(y_true)
    return acc

def fit_theta(x, y, theta, alpha, learning_rate, iters_num = 15000, cost_checking_time = 100, adagrad = True):
    N, V = x.shape
    G = np.zeros(V)
    eps = 0.00000001
    reminder = 0
    cost_prev = cost(x, y, theta, alpha)
    
    for i in range(iters_num):
        gradient = grad(x, y, theta, alpha)
        if adagrad:
            G += gradient ** 2
            eta = learning_rate / np.sqrt(G + eps)
            theta -= eta * gradient
        else:
            theta -= learning_rate * gradient

        reminder += 1
        if reminder == cost_checking_time:
            cost_curr = cost(x, y, theta, alpha)           
            if abs(cost_prev - cost_curr) < 0.0001:
                break
            cost_prev = cost_curr
            reminder = 0

    print('Number of iterations:', i)
    return theta

def init_theta(size):
    return np.zeros(size)

def cost(x, y, theta, alpha):
    N = len(y)
    eps = 0.00000001
    reg_contrib = np.sum(theta[1:] ** 2 * alpha)
    preds = sigmoid(x.dot(theta))
    npreds = np.log(1 - preds + eps)
    preds = np.log(preds + eps)
    ny = 1 - y
    return (-1 / N) * (y.dot(preds) + ny.dot(npreds)) + reg_contrib

def grad(x, y, theta, alpha):
    N = len(y)
    gradient = x.transpose().dot(sigmoid(x.dot(theta)) - y) / N
    reg_grad = theta * (alpha * 2)
    reg_grad[0] = 0
    gradient += reg_grad
    return gradient

def fit_params(x, y, alphas, learning_rates):
    print('Fitting hyperparameters ...')
    N, V = x.shape
    rows = np.random.permutation(N)
    delimiter = int(np.ceil(0.8 * N))

    x_train = x[rows[:delimiter], :]
    x_dev = x[rows[delimiter:], :]
    y_train = y[rows[:delimiter]]
    y_dev = y[rows[delimiter:]]

    max_acc = 0.0
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
    with open('params.txt', 'a') as outp:
        print('(alpha, learning_rate)', params, 'accuracy:', max_acc, file = outp)
    return params

def make_words_map(unique_words):
    words_map = {}
    for i in range(len(unique_words)):
        words_map[unique_words[i]] = i
    return words_map

def make_vocabulary(texts):
    vocab = defaultdict(int)
    for text in texts:
        for word in text:
            vocab[word] = 1
    return vocab

def classify(texts, params):
    texts = init_texts(texts)
    theta = params['theta']
    words_map = params['words_map']
    N, V = params['shape']

    data = []
    indices = []
    indptr = [0]

    for text in texts:
        text = set(text)
        indices.append(0)
        data.append(1)
        for word in text:
            try:
                idx = words_map[word]
                indices.append(idx + 1)
                data.append(1)
            except KeyError:
                continue
        indptr.append(len(indices))

    x = csr.csr_matrix((data, indices, indptr), dtype = float, shape = (len(texts), V))

    labels = list(map(lambda x: 'pos' if x >= 0 else 'neg', x.dot(theta)))

    return labels

def train(texts, labels, alpha = 5e-06, learning_rate = 0.04, enable_params_fitting = False):
    print('alpha, learning_rate:', alpha, learning_rate)
    texts = init_texts(texts)
    big_vocab = make_vocabulary(texts)
    unique_words = np.array([word for word in big_vocab])
    words_map = make_words_map(unique_words)
    alphas = [5e-05, 5e-06, 5e-07]
    learning_rates = [0.01, 0.1, 0.05, 0.005]
    data = []
    indices = []
    indptr = [0]

    for text in texts:
        text = set(text)
        indices.append(0)
        data.append(1)
        for word in text:
            try:
                idx = words_map[word]
                indices.append(idx + 1)
                data.append(1)
            except KeyError:
                continue
        indptr.append(len(indices))

    x = csr.csr_matrix((data, indices, indptr), dtype = float)
    y = np.array(list(map(lambda x: 1 if x == 'pos' else 0, labels)))

    if enable_params_fitting:
        alpha, learning_rate = fit_params(x, y, alphas, learning_rates)
        
    theta = init_theta(len(words_map) + 1)
    theta = fit_theta(x, y, theta, alpha, learning_rate)

    res = {}
    res['theta'] = theta
    res['words_map'] = words_map
    res['shape'] = x.shape
    return res