import re
import numpy as np
import scipy.sparse as csr
from collections import defaultdict

def initTexts(texts, maxNGramSize = 1):
    texts = [tokenizer(text) for text in texts]
    for i in range(len(texts)):
        nGrams = []
        for j in range(maxNGramSize - 1):
            nGrams += addNGrams(texts[i], j + 2)
        texts[i] += nGrams
    return texts

def addNGrams(tokens, n):
    buff = []
    for i in range(len(tokens) - n + 1):
        ngram = ''
        for j in range(n):
            ngram += (tokens[i + j] + ' ')
        buff.append(ngram.strip())
    return buff

def tokenizer(text, toLower = True, withStopWords = False):
    if withStopWords:
        with open("Starter code/stopwords.txt", "r") as f:
           stopwords = map(lambda x: x.strip(), f.readlines())
    else:
        stopwords = []
    buff = []
    stopwords = []
    if toLower:
        text = text.lower()
    text = re.findall('\w+', text)
    for word in text:
        if not word in stopwords:
            buff.append(word)
    return buff

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def accuracy(preds, trues):
    correct = sum(y1 == y2 for y1, y2 in zip(preds, trues))
    acc = 100.0 * correct / len(trues)
    return acc

def fitTheta(x, y, theta, alpha, learningRate = 0.04, itersNum = 15000, timeToCheckCost = 100, adagrad = True, convCoeff = 0.00001, devToo = False, xDev = None, yDev = None):
    thetaSize = len(theta)
    eps = 1e-08
    beta1 = 0.9
    beta2 = 0.999
    M = np.zeros(thetaSize)
    V = np.zeros(thetaSize)
    G = np.zeros(thetaSize)
    reminder = 0
    if devToo:
        costs = [(cost(x, y, theta, alpha), cost(xDev, yDev, theta, 0))]
    else:
        costs = [(cost(x, y, theta, alpha), None)]
    for i in range(itersNum):
        try:
            gradient = grad(x, y, theta, alpha)
            if adagrad:
                G += gradient ** 2
                eta = learningRate / np.sqrt(G + eps)
                theta -= eta * gradient
            else:
                M = beta1 * M + (1 - beta1) * gradient
                V = beta2 * V + (1 - beta2) * np.power(gradient, 2)
                m = M / (1 - beta1 ** (i + 1))
                v = V / (1 - beta2 ** (i + 1))
                theta -= learningRate * (m / (np.power(v, 0.5) + eps))

            if reminder == timeToCheckCost:
                currCost = cost(x, y, theta, alpha)
                if not devToo:
                    costs = [(currCost, None)] + costs
                else:
                    costs = [(currCost, cost(xDev, yDev, theta, 0))] + costs
                #print(currCost)
                prevCost, _ = costs[1]
                if abs(prevCost - currCost) < convCoeff:
                    break
                reminder = 0
            reminder += 1
        except KeyboardInterrupt:
            break

    print('Number of iterations:', i)
    return theta, costs

def initTheta(size):
    return np.zeros(size)

def cost(x, y, theta, alpha):
    N = len(y)
    eps = 1e-08
    regCost = np.sum(theta[1:] ** 2 * alpha)
    preds = sigmoid(x.dot(theta))
    nPreds = np.log(1 - preds + eps)
    preds = np.log(preds + eps)
    nY = 1 - y
    return (-1 / N) * (y.dot(preds) + nY.dot(nPreds)) + regCost

def grad(x, y, theta, alpha):
    N = len(y)
    gradient = x.transpose().dot(sigmoid(x.dot(theta)) - y) / N
    regGrad = theta * (alpha * 2)
    regGrad[0] = 0
    gradient += regGrad
    return gradient

def getIndices(labels, label):
    indices = []
    for i in range(len(labels)):
        if labels[i] == label:
            indices.append(i)
    return indices

def calculateProbabilities(texts, wordsMap):
    numTexts = len(texts)
    vocabSize = len(wordsMap)
    wordsProbs = np.zeros(vocabSize)
    for text in texts:
        tmp = np.zeros(vocabSize)
        for word in text:
            try:
                idx = wordsMap[word]
                tmp[idx] = 1
            except KeyError:
                continue
        wordsProbs += tmp
    wordsProbs += 1
    wordsProbs /= (numTexts + 2)    
    return wordsProbs

def makeWordsMap(texts, labels, vocabSize = 500000):
    wordsMap = {}
    bigVocab = makeVocab(texts)
    uniqueWords = np.array(sorted(bigVocab, key=bigVocab.get, reverse=True)[:vocabSize])
    for i in range(len(uniqueWords)):
        wordsMap[uniqueWords[i]] = i
    posIndices = getIndices(labels, 'pos')
    negIndices = getIndices(labels, 'neg')
    posTexts = [texts[i] for i in posIndices]
    negTexts = [texts[i] for i in negIndices]
    posProbs = calculateProbabilities(posTexts, wordsMap)
    negProbs = calculateProbabilities(negTexts, wordsMap)
    bayesWeights = np.log(posProbs / negProbs)
    for word in wordsMap:
        key = wordsMap[word]
        wordsMap[word] = (key, bayesWeights[key])
    return wordsMap

def texts2x(texts, labels = None, wordsMap = None, shape = None, vocabSize = 500000, maxNGramSize = 3, weighted = True):
    texts = initTexts(texts, maxNGramSize = maxNGramSize)
    if wordsMap == None:
        if labels == None:
            raise RuntimeError
        wordsMap = makeWordsMap(texts, labels, vocabSize = vocabSize)
        if not weighted:
            for word in wordsMap:
                idx, _ = wordsMap[word]
                wordsMap[word] = (idx, 1)
    data = []
    indices = []
    indptr = [0]

    for text in texts:
        text = set(text)
        indices.append(0)
        data.append(1)
        for word in text:
            try:
                idx, weight = wordsMap[word]
                indices.append(idx + 1)
                data.append(weight)
            except KeyError:
                continue
        indptr.append(len(indices))
    if shape == None:
        x = csr.csr_matrix((data, indices, indptr), dtype = float)
    else:
        _, V = shape
        x = csr.csr_matrix((data, indices, indptr), dtype = float, shape = (len(texts), V))
    return (x, wordsMap)

def makeVocab(texts):
    vocab = defaultdict(int)
    for text in texts:
        for word in text:
            vocab[word] += 1
    return dict(vocab)

def classify(texts, params):
    theta = params['theta']
    wordsMap = params['wordsMap']
    shape = params['shape']
    maxNGramSize = params['maxNGramSize']
    x, _ = texts2x(texts, wordsMap = wordsMap, shape = shape, maxNGramSize = maxNGramSize)
    labels = list(map(lambda x: 'pos' if x >= 0 else 'neg', x.dot(theta)))
    return labels

def train(texts, labels, alpha = 1e-05, learningRate = 0.04, maxNGramSize = 3):
    x, wordsMap = texts2x(texts, labels, maxNGramSize = maxNGramSize)
    _, V = x.shape
    y = np.array(list(map(lambda x: 1 if x == 'pos' else 0, labels)))
    theta = initTheta(V)
    theta, costs = fitTheta(x, y, theta, alpha, learningRate)
    res = {}
    res['theta'] = theta
    res['wordsMap'] = wordsMap
    res['shape'] = x.shape
    res['maxNGramSize'] = maxNGramSize
    return (res, costs)

def textsWithErrors(preds, labels, texts):
    errors = []
    posErrors = []
    negErrors = []
    for i in range(len(labels)):
        if preds[i] != labels[i]:
            errors.append(texts[i])
            if labels[i] == 'pos':
                posErrors.append(texts[i])
            else:
                negErrors.append(texts[i])
    return (errors, posErrors, negErrors)
