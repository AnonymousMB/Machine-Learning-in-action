# coding=utf-8
from numpy import *
from matplotlib import pyplot as plt
import math


class Data():
    total = None  # 样本总数
    featureNum = None  # 特征数量

    def __init__(self, weight, num, feature, label):
        self.weight = weight  # 权值
        self.num = num  # 序号
        self.feature = feature  # 输入特征
        self.label = label  # 类别标签
        self.tempWeight = None


class Tree():
    def __init__(self):
        self.leftLabel = None
        self.rightLabel = None
        # 左右标签
        self.weight = None
        # 分类器权重
        self.splitFeatureValue = None
        # 用于划分的值
        self.splitFeatureOrder = None
        # 用于划分的属性序号
        self.err = None
        # 基学习器在训练数据集上的分类误差率

    def classifier(self, feature):
        # 基学习器的分类器
        """
        :type feature: matrix
        """
        value = feature[0, self.splitFeatureOrder]
        if value > self.splitFeatureValue:
            return self.rightLabel
        else:
            return self.leftLabel


class Classifier():
    def __init__(self, treeSet):
        self.treeSet = treeSet

    def classify(self, feature):
        sumResult = 0
        for tree in self.treeSet:
            value = tree.classifier(feature)
            sumResult += tree.weight * value
        return sumResult


def loadDataSet(path):
    fr = open(path, 'r')
    dataSet = []
    labelSet = []
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataSet.append(lineArr[0:-1])
        labelSet.append(lineArr[-1])
    dataMat = mat(dataSet, float)
    labelMat = mat(labelSet, float)
    count = 0
    m, n = shape(dataMat)
    Data.featureNum = n
    Data.total = m
    dataSet = []
    weight = float(1.0 / m)
    # 初始化权值分布
    for count in range(m):
        if labelMat[0, count] != 1:
            labelMat[0, count] = -1
        feature = dataMat[count, :]
        data = Data(weight, count, feature, labelMat[0, count])
        dataSet.append(data)
        count += 1
    dataSet = mat(dataSet)
    return dataSet
    # dataSet是一个 1*（训练样本数）的矩阵


def loadSimpleData():
    path1 = r'data/plrx/simpleTest.txt'
    path2 = r'data/mlia/Ch06/testSetRBF.txt'
    return loadDataSet(path2)


def chooseSplitValue(dataSet, featureOrder):
    # 选择划分值
    labelList = []
    i = 0
    err = inf
    # 初始化误差为正无穷
    leftLabel = 0
    rightLabel = 0
    # 初始化左标签和右标签
    featureInf = {"err": 0, "order": 0, "value": 0, "labelList": None}
    # 初始化划分值
    for i in range(2):
        # 分别检测左标签为-1右标签为1，左标签为1右标签为-1两种情况
        leftLabelTemp = float(-1 + i * 2)
        rightLabelTemp = -leftLabelTemp
        dataNum = 0
        for dataNum in range(Data.total):
            # 将每个样本的特征值作为划分值计算误差
            count = 0
            left = set()
            right = set()
            for count in range(Data.total):
                # 根据dataNum对应样本在对应属性上的对应值划分左右集合，集合内容为样本序号
                data = dataSet[0, count]
                if data.feature[0, featureOrder] <= dataSet[0, dataNum].feature[0, featureOrder]:
                    left.add(data.num)
                else:
                    right.add(data.num)
                count += 1
            leftErr = 0
            count = 0
            for count in left:
                # 计算左子集误差和
                data = dataSet[0, count]
                if data.label != leftLabelTemp:
                    leftErr += data.weight
                count += 1
            count = 0
            rightErr = 0
            for count in right:
                # 计算右子集误差和
                data = dataSet[0, count]
                if data.label != rightLabelTemp:
                    rightErr += data.weight
                count += 1
            if leftErr + rightErr < err:
                # 如果当前误差小于最小误差
                err = leftErr + rightErr
                leftLabel = leftLabelTemp
                rightLabel = rightLabelTemp
                featureInf['value'] = dataSet[0, dataNum].feature[0, featureOrder]
            dataNum += 1
        i += 1
    labelList.append(leftLabel)
    labelList.append(rightLabel)
    featureInf['err'] = err
    featureInf['order'] = featureOrder
    featureInf['labelList'] = labelList
    return featureInf


def chooseFeature(dataSet):
    # 挑选划分属性
    stdList = []
    count = 0
    err = inf
    featureInf = None
    for count in range(Data.featureNum):
        featureInfTemp = chooseSplitValue(dataSet, count)
        if featureInfTemp['err'] < err:
            err = featureInfTemp['err']
            featureInf = featureInfTemp
        count += 1
    return featureInf


def treeGenerate(dataSet):
    # 生成决策树
    tree = Tree()
    featureInf = chooseFeature(dataSet)
    print featureInf
    labelList = featureInf['labelList']
    tree.leftLabel = labelList[0]
    tree.rightLabel = labelList[1]
    if featureInf['err'] != 0:
        tree.weight = 0.5 * math.log((1.0 - featureInf['err']) / featureInf['err'])
    else:
        tree.weight = 1
    tree.splitFeatureOrder = featureInf['order']
    tree.splitFeatureValue = featureInf['value']
    tree.err = featureInf['err']
    return tree


def updateWeight(dataSet, tree):
    i = 0
    Z = 0
    for i in range(Data.total):
        data = dataSet[0, i]
        Z += data.weight * exp(-tree.weight * data.label * tree.classifier(data.feature))
        i += 1
    i = 0
    for i in range(Data.total):
        data = dataSet[0, i]
        data.weight = data.weight * exp(-tree.weight * data.label * tree.classifier(data.feature)) / Z
        i += 1
    return dataSet


def terminate(classifier, dataSet, minErr):
    total = 0
    correctCount = 0
    for i in range(Data.total):
        data = dataSet[0, i]
        value = classifier.classify(data.feature)
        if value >= 0:
            value = 1
        else:
            value = -1
        if value == data.label:
            correctCount += 1
        total += 1
    correctRate = float(correctCount) / total
    print "迭代完成，正确率为", correctRate
    if 1 - correctRate <= minErr:
        return False
    else:
        return True


def adaBoost(dataSet, iterNum, minErr):
    # AdaBoost主体函数
    count = 0
    classifier = None
    treeSet = set()
    flag = True
    # 运行条件初始化为真
    while (count < iterNum and flag is True):
        tree = treeGenerate(dataSet)
        # 训练一个决策树桩
        dataSet = updateWeight(dataSet, tree)
        # 更新权值分布
        treeSet.add(tree)
        # 添加一棵树进树集合
        classifier = Classifier(treeSet)
        # 构造分类器
        flag = terminate(classifier, dataSet, minErr)
        # 检测停机条件
        count += 1
    if flag is True:
        print "迭代超过最大次数，终止迭代并返回分类器"
    return classifier


def draw(dataSet, classifier):
    # 绘图函数
    fig, ax = plt.subplots(1, 1)
    i = 0
    for i in range(Data.total):
        data = dataSet[0, i]
        if data.label != 1:
            ax.scatter(data.feature[0, 0], data.feature[0, 1], s=20, marker='o', c='g')
        else:
            ax.scatter(data.feature[0, 0], data.feature[0, 1], s=20, marker='o', c='b')
        i += 1
    for tree in classifier.treeSet:
        axis = tree.splitFeatureOrder
        value = tree.splitFeatureValue
        otherAxis = arange(-1, 1, 0.1)
        if axis == 0:
            value = otherAxis * 0 + value + 0.1
            ax.plot(value, otherAxis)
        else:
            value = otherAxis * 0 + value + 0.1
            ax.plot(otherAxis, value)
    plt.show()

dataSet = loadSimpleData()
classifier = adaBoost(dataSet, 100, 0)
draw(dataSet, classifier)