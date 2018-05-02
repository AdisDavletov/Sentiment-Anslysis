import os
import re
import numpy as np
from collections import defaultdict

def tokenizer(string, toLower = True):
    with open("../stopwords.txt", "r") as f:
        stopwords = map(lambda x: x.strip(), f.readlines())
    if toLower:
        string = string.lower()
    tokens = re.findall('\w+', string)
    res = []
    for token in tokens:
        if token not in stopwords:
            res.append(token)
    return res

def makeVocabulary(texts):
    vocabulary = defaultdict(int)
    for text in texts:
        for word in text:
            vocabulary[word] += 1
    return vocabulary

def addNGrams(tokens, n):
    buff = []
    for i in range(len(tokens) - n + 1):
        ngram = ''
        for j in range(n):
            ngram += (tokens[i + j] + ' ')
        buff.append(ngram.strip())
    return buff

def getIndices(labels, label):
    indices = []
    for i in range(len(labels)):
        if labels[i] == label:
            indices.append(i)
    return indices

def calculateProbabilitiesBnb(texts, wordsMap):
    numberOfTexts = len(texts)
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
    wordsProbs /= (numberOfTexts + 2)    
    return wordsProbs

def calculateProbabilitiesMnb(texts, wordsMap, alpha = 1):
    number_of_words = 0
    vocabSize = len(wordsMap)
    wordsProbs = np.zeros(vocabSize)
    for text in texts:
        tmp = np.zeros(vocabSize)
        for word in text:
            try:
                idx = wordsMap[word]
                number_of_words += 1
                tmp[idx] += 1
            except KeyError:
                tmp[vocabSize - 1] += 1
                continue
        wordsProbs += tmp
    wordsProbs += alpha
    wordsProbs /= (number_of_words + vocabSize * alpha)
    return wordsProbs

def initTexts(texts):
    texts = [tokenizer(string) for string in texts]
    texts = [unigrams + addNGrams(unigrams, 2) for unigrams in texts]
    return texts

def makeVocabulary(texts):
    vocab = defaultdict(int)
    for text in texts:
        for word in text:
            vocab[word] += 1
    return vocab

def makeWordsMap(uniqueWords):
    wordsMap = {}
    for i in range(len(uniqueWords)):
        wordsMap[uniqueWords[i]] = i
    return wordsMap

def train(texts, labels, mode = "bernoulli"):
    if mode == 'bernoulli':
        primaryVocabSize   = 70000
        secondaryVocabSize = 10200
    else:
        primaryVocabSize   = 26000
        secondaryVocabSize = 2700

    texts = initTexts(texts)
    big_vocab = makeVocabulary(texts)
    uniqueWords = np.array(sorted(big_vocab, key = big_vocab.get, reverse = True)[:primaryVocabSize] + ['UNK'])    
    wordsMap = makeWordsMap(uniqueWords)

    posIndices = getIndices(labels, 'pos')
    negIndices = getIndices(labels, 'neg')
    posTexts = [texts[i] for i in posIndices]
    negTexts = [texts[i] for i in negIndices]

    if mode == "bernoulli":
        posProbs = calculateProbabilitiesBnb(posTexts, wordsMap)
        negProbs = calculateProbabilitiesBnb(negTexts, wordsMap)
    else:
        posProbs = calculateProbabilitiesMnb(posTexts, wordsMap)
        negProbs = calculateProbabilitiesMnb(negTexts, wordsMap)

    posClassProb = len(posTexts) / len(texts)
    negClassProb = len(negTexts) / len(texts)
    
    bayesWeights = posProbs / negProbs
    sorted_indices = bayesWeights.argsort()
    
    new_unique_words = np.array(list(uniqueWords[sorted_indices[:secondaryVocabSize]]) + list(uniqueWords[sorted_indices[(len(sorted_indices) - secondaryVocabSize):]]) + ['UNK'])
    wordsMap = makeWordsMap(new_unique_words)
    if mode == "bernoulli":
        posProbs = calculateProbabilitiesBnb(posTexts, wordsMap)
        negProbs = calculateProbabilitiesBnb(negTexts, wordsMap)
    else:
        posProbs = calculateProbabilitiesMnb(posTexts, wordsMap)
        negProbs = calculateProbabilitiesMnb(negTexts, wordsMap)
    params = {}
    params['pos'] = posProbs
    params['neg'] = negProbs
    params['posClassProb'] = posClassProb
    params['negClassProb'] = negClassProb
    params['wordsMap'] = wordsMap
    return params

def classify(texts, params, mode = "bernoulli"):
    texts = initTexts(texts)
    posProbs = params['pos']
    negProbs = params['neg']
    posClassProb = params['posClassProb']
    negClassProb = params['negClassProb']
    wordsMap = params['wordsMap']
    vocabSize = len(wordsMap)

    if mode == "bernoulli":
        labels = classifyBnb(posProbs, negProbs, posClassProb, negClassProb, wordsMap, texts)
    else:
        labels = classifyMnb(posProbs, negProbs, posClassProb, negClassProb, wordsMap, texts)
    
    return labels
    

def classifyBnb(posProbs, negProbs, posClassProb, negClassProb, wordsMap, texts):
    nPosProbs = 1 - posProbs
    nNegProbs = 1 - negProbs
    vocabSize = len(wordsMap)
    labels = []

    for text in texts:
        b = np.zeros(vocabSize)
        for word in text:
            try:
                b[wordsMap[word]] = 1
            except KeyError:
                b[vocabSize - 1] = 1
                continue
        nb = 1 - b

        posRes = sum(np.log(posProbs * b + nPosProbs * nb)) + np.log(posClassProb)
        negRes = sum(np.log(negProbs * b + nNegProbs * nb)) + np.log(negClassProb)

        if np.exp(posRes) > np.exp(negRes):
            labels.append('pos')
        else:
            labels.append('neg')
            
    return labels

def classifyMnb(posProbs, negProbs, posClassProb, negClassProb, wordsMap, texts):
    vocabSize = len(wordsMap)
    labels = []
    for text in texts:
        indices = np.zeros(vocabSize)
        for word in text:
            try:
                idx = wordsMap[word]
                indices[idx] += 1
            except KeyError:
                indices[vocabSize - 1] = 1
                continue
        posRes = np.sum(np.log(posProbs) * indices) + np.log(posClassProb)
        negRes = np.sum(np.log(negProbs) * indices) + np.log(negClassProb)
        if np.exp(posRes) > np.exp(negRes):
            labels.append('pos')
        else:
            labels.append('neg')
    return labels

def accuracy(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    correct = sum(y1 == y2 for y1, y2 in zip(y_pred, y_true))
    acc = 100 * correct / len(y_true)
    return acc

def textsWithErrors(preds, labels, texts):
    errors = []
    for i in range(len(labels)):
        if preds[i] != labels[i]:
            errors.append(texts[i])
    return errors