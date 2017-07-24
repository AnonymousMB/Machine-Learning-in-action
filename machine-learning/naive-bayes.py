# coding=utf-8
from numpy import *
import os
import re


def loadVocabulary():
    pattern = re.compile(r'[^a-zA-Z]')
    filePath = r"data\mlia\Ch04"
    spamHamSwitcher = [r"\ham", r"\spam"]
    vocabulary = []
    for i in range(2):  # 因为在Ch04提供的训练集中，spam和ham邮件分装在两个不同的文件夹
        for file in os.listdir(filePath + spamHamSwitcher[i]):  # 遍历训练集中邮件所在的文件夹中的所有文件
            fr = open(filePath + spamHamSwitcher[i] + r"/" + file, "r")  # 以只读模式打开一个样本
            for line in fr.readlines():
                wordArr = line.strip().split()
                for word in wordArr:
                    word = re.sub(pattern, "", word)
                    if word is not "":
                        if word not in vocabulary:
                            vocabulary.append(word)  # 扩充词典
    return vocabulary


def loadDataSet(vocabulary):
    pattern = re.compile(r"[^a-zA-Z]")
    filePath = r"data\mlia\Ch04"
    spamHamSwitcher = [r"\ham", r"\spam"]
    hamMat = []
    spamMat = []
    for i in range(2):
        for file in os.listdir(filePath + spamHamSwitcher[i]):
            fr = open(filePath + spamHamSwitcher[i] + r"/" + file, "r")
            hamVoc = zeros(len(vocabulary))
            spamVoc = zeros(len(vocabulary))
            for line in fr.readlines():
                lineArr = line.strip().split()
                for word in lineArr:
                    word = re.sub(pattern, "", word)
                    if word in vocabulary:
                        if i == 0:
                            hamVoc[vocabulary.index(word)] = 1
                        else:
                            spamVoc[vocabulary.index(word)] = 1
            if i == 0:
                hamMat.append(hamVoc)
            else:
                spamMat.append(spamVoc)
    return hamMat, spamMat


def naiveBayes(hamMat, spamMat):
    hamCount, n1 = shape(hamMat)  # 获取正常邮件数量
    spamCount, n2 = shape(spamMat)  # 获取垃圾邮件数量
    totalCount = hamCount + spamCount  # 获取邮件总数
    weights = zeros((len(hamMat[0]), len(spamMat[0]), 1))  # 创建参数向量组
    for j in range(len(hamMat[0])):  # 对于词典中的每一个词
        for i in range(hamCount):  # 对于每一个正常邮件
            weights[0][j] = weights[0][j] + hamMat[i][j]
        weights[0][j] = (float(weights[0][j]) + 1) / (hamCount + 2)  # 计算参数Φ1
        for i in range(spamCount):
            weights[1][j] = weights[1][j] + spamMat[i][j]
        weights[1][j] = (float(weights[1][j]) + 1) / (spamCount + 2)  # 计算参数Φ0
    weights[2][0] = float(spamCount) / totalCount  # 计算参数Φy
    return weights


def classification(weights, target, vocabulary):
    hamRate = 1
    spamRate = 1
    pattern = re.compile(r"[^a-zA-Z]")
    length = len(vocabulary)  # 获取词典长度
    fileVoc = zeros(length)
    for line in target.readlines():  # 将测试邮件内容解析为特征向量
        wordArr = line.strip().split()
        for word in wordArr:
            word = re.sub(pattern, "", word)
            if word in vocabulary:
                fileVoc[vocabulary.index(word)] = 1
    for i in range(length):# 计算如果该邮件是正常邮件，那么它的特征为正常邮件特征的概率
        temp = float(weights[0][i] * fileVoc[i])
        if temp != 0:
            hamRate = float(hamRate * temp)
    for i in range(length):# 计算如果该邮件是垃圾邮件，那么它的特征为垃圾邮件特征的概率
        temp = float(weights[1][i] * fileVoc[i])
        if temp != 0:
            spamRate = float(spamRate * temp)
    result = (spamRate * weights[2][0]) / (hamRate * (1 - weights[2][0]) + spamRate * weights[2][0])
    # 计算得到该邮件是垃圾邮件的概率
    return result


def test(classification, weights, vocabulary):
    pattern = re.compile(r'[^a-zA-Z]')
    filePath = r"data\mlia\Ch04"
    spamHamSwitcher = [r"\ham", r"\spam"]
    error = 0
    for i in range(2):
        for file in os.listdir(filePath + spamHamSwitcher[i]):
            fr = open(filePath + spamHamSwitcher[i] + r"/" + file, "r")
            error = error + square(float(i - classification(weights, fr, vocabulary)))
    return error

vocabulary = loadVocabulary()
hamMat, spamMat = loadDataSet(vocabulary)
weights = naiveBayes(hamMat, spamMat)
error = test(classification,weights,vocabulary)

fr=open(r'test.txt')
result=classification(weights,fr,vocabulary)
if result>0.5:
    print "spam"
else:
    print "ham"