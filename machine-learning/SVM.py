# coding=utf-8
from numpy import *
from matplotlib import pyplot as plt
import pandas as pd


class alphaClass(object):
    def __init__(self):
        self.alpha = 0
        self.num = 0


def loadDataSet(path):
    # 数据集读取函数
    fr = open(path, 'r')
    dataSet = []
    for line in fr.readlines():
        lineArr = line.strip().split()
        data = []
        i = 0
        for attr in lineArr:
            lineArr[i] = float(lineArr[i])
            data.append(lineArr[i])
            i += 1
        if data[-1] == 2:
            data[-1] = -1
        dataSet.append(data)
    dataSet = mat(dataSet)
    return dataSet


def kernel(xi, xj):
    # 高斯径向核函数
    """
    :type theta: int
    :type xj: numpy.matrixlib.defmatrix.matrix
    :type xi: numpy.matrixlib.defmatrix.matrix
    """
    theta = sqrt(2)
    vector = xi - xj
    output = exp(-float((vector * vector.transpose())[0, 0]) / (2 * theta * theta))
    return output


def hypothesis(xi, dataSet, alphas, b):
    # 输出函数
    sumResult = 0
    j = 0
    for data in dataSet:
        sumResult += float(alphas[j]) * data[0, -1] * kernel(xi, data)
        j += 1
    return b + sumResult


def searchAlphaOne(alphas, dataSet, C, b, alphaExceptionNumSet):
    # 搜寻第一个变量的外层循环函数
    alphaBlock = alphaClass()
    alpha = 0
    alphaNum = 0
    count = 0
    deviation = 0
    for alphaPre in alphas:
        if alphaPre in alphaExceptionNumSet:
            count += 1
            continue
        g = hypothesis(dataSet[count], dataSet, alphas, b)
        condition = dataSet[count, -1] * g
        if alphaPre == 0:
            if condition < 1:
                if deviation < abs(condition - 1):
                    deviation = abs(condition - 1)
                    alpha = alphaPre
                    alphaNum = count
        elif alphaPre > 0 and alphaPre < C:
            if condition != 1:
                if deviation < abs(condition - 1):
                    deviation = abs(condition - 1)
                    alpha = alphaPre
                    alphaNum = count
        elif alphaPre == C:
            if condition > 1:
                if deviation < abs(condition - 1):
                    deviation = abs(condition - 1)
                    alpha = alphaPre
                    alphaNum = count
        elif alphaPre > C:
            print "alphas中出现了异常变量，该变量大于C"
            return
        elif alphaPre < 0:
            print "alphas中出现了异常变量，该变量小于0"
            return
        count += 1
    alphaBlock.alpha = alpha
    alphaBlock.num = alphaNum
    return alphaBlock


def EUpdater(EList, alphas, bnew, dataSet, C):
    # 误差E更新函数
    count = 0
    for data in dataSet:
        j = 0
        sumResult = 0.0
        for alpha in alphas:
            if alpha > 0 and alpha < C:
                sumResult += data[0, -1] * alpha * kernel(data, dataSet[j])
            j += 1
        EList[count] = sumResult + bnew - data[0, -1]
        count += 1


def alphasUpdate(alphaBlockOne, alphaBlockTwo, alphas):
    # alpha集合更新函数
    alphas[alphaBlockOne.num] = alphaBlockOne.alpha
    alphas[alphaBlockTwo.num] = alphaBlockTwo.alpha


def searchAlphaTwo(alphaBlockOne, EList, alphas, dataSet):
    # 搜寻第二个变量的内层循环函数
    """
    :type dataSet: numpy.matrixlib.defmatrix.matrix
    :type alphas: numpy.matrixlib.defmatrix.matrix
    :type: EList: list
    :type alphaBlockOne: alphaClass
    """
    alphaBlock = alphaClass()
    alpha = 0.0
    count = 0
    alphaNum = 0
    weight = 0
    for alphaPre in alphas:
        if count == alphaBlockOne.num:
            count += 1
            continue
        weightPre = abs(EList[count] - EList[alphaBlockOne.num])
        if weight < weightPre:
            weight = weightPre
            alpha = alphaPre
            alphaNum = count
        count += 1
    alphaBlock.num = alphaNum
    alphaBlock.alpha = alpha
    return alphaBlock


def ECalc(num, dataSet, EList, alphas, b):
    # 误差E计算函数
    E = hypothesis(dataSet[num], dataSet, alphas, b) - dataSet[num, -1]
    EList[num] = E
    return E


def bUpdater(alphaBlockOne, alphaBlockTwo, alphas, dataSet, EList, bOld, C):
    # 阈值b更新函数
    num1 = alphaBlockOne.num
    alpha1 = alphaBlockOne.alpha
    num2 = alphaBlockTwo.num
    alpha2 = alphaBlockTwo.alpha
    bnew1 = -EList[num1] - dataSet[num1, -1] * kernel(dataSet[num1], dataSet[num1]) * (alpha1 - alphas[num1]) \
            - dataSet[num2, -1] * kernel(dataSet[num2], dataSet[num1]) * (alpha2 - alphas[num2]) + bOld
    bnew2 = -EList[num2] - dataSet[num1, -1] * kernel(dataSet[num1], dataSet[num2]) * (alpha1 - alphas[num1]) \
            - dataSet[num2, -1] * kernel(dataSet[num2], dataSet[num2]) * (alpha2 - alphas[num2]) + bOld
    if (alpha1 > 0 and alpha1 < C) and (alpha2 > 0 and alpha2 < C):
        return bnew1
    if (alpha1 == 0 or alpha1 == C) or (alpha2 == 0 or alpha2 == C):
        return float(bnew1 + bnew2) / 2
    print "出现异常变量"
    return None


def bCalc(alphas, dataSet, C):
    # 阈值b计算函数
    b = 0
    count = 0
    for alphaPre in alphas:
        if alphaPre > 0 and alphaPre < C:
            sumResult = 0
            i = 0
            for data in dataSet:
                sumResult += dataSet[i, -1] * alphas[i] * kernel(data, dataSet[count])
                i += 1
            b = dataSet[count, -1] - sumResult
            return b
        count += 1
    return b


def alphaCutter(alpha2, alphaBlockTwo, alphaBlockOne, alphas, C, dataSet):
    num1 = alphaBlockOne.num
    num2 = alphaBlockTwo.num
    L = 0
    H = C
    if dataSet[num1, -1] != dataSet[num2, -1]:
        L = max(0, alphas[num2] - alphas[num1])
        H = min(C, C + alphas[num2] - alphas[num1])
    else:
        L = max(0, alphas[num2] + alphas[num1] - C)
        H = min(C, alphas[num2] + alphas[num1])
    if alpha2 > H:
        return H
    elif alpha2 < L:
        return L
    else:
        return alpha2


def alphaSolver(alphaBlockOne, alphaBlockTwo, EList, alphas, dataSet, C):
    # 两个变量的凸二次规划问题优化函数
    alphaBlockNew1 = alphaClass()
    alphaBlockNew2 = alphaClass()
    num1 = alphaBlockOne.num
    num2 = alphaBlockTwo.num
    eta = kernel(dataSet[num1], dataSet[num1]) + kernel(dataSet[num2], dataSet[num2]) \
          - 2 * kernel(dataSet[num1], dataSet[num2])
    alpha2 = alphaBlockTwo.alpha + dataSet[num2, -1] * (EList[num1] - EList[num2]) / eta
    alpha2 = alphaCutter(alpha2, alphaBlockTwo, alphaBlockOne, alphas, C, dataSet)
    # 剪辑alpha2
    alpha1 = alphaBlockOne.alpha + dataSet[num1, -1] * dataSet[num2, -1] * (alphaBlockTwo.alpha - alpha2)
    alphaBlockNew1.alpha = alpha1
    alphaBlockNew1.num = num1
    alphaBlockNew2.alpha = alpha2
    alphaBlockNew2.num = num2
    return alphaBlockNew1, alphaBlockNew2


def signFunction(x, dataSet, alphas, b):
    # 决策符号函数
    result = hypothesis(x, dataSet, alphas, b)
    if result > 0:
        return 1
    else:
        return -1


def terminateCondition(alphas, dataSet, b, C, accuracy):
    # 停机条件判断函数
    flag = 1
    i = 0
    for alpha in alphas:
        condition = dataSet[i, -1] * hypothesis(dataSet[i], dataSet, alphas, b)
        if alpha == 0:
            if condition < 1:
                flag = 0
                continue
        elif alpha > 0 and alpha < C:
            if condition != 1:
                flag = 0
                continue
        elif alpha == C:
            if condition > 1:
                flag = 0
                continue
        elif alpha > C:
            print "alphas中出现了异常变量，该变量大于C"
            flag = 0
        elif alpha < 0:
            print "alphas中出现了异常变量，该变量小于0"
            flag = 0
        i += 1
    if flag == 1:
        return 1
    # 此时flag=0
    rate = {"correct": 0, "total": 0}
    i = 0
    for data in dataSet:
        rate['total'] += 1
        result = data[0, -1] * signFunction(data, dataSet, alphas, b)
        if result > 0:
            rate['correct'] += 1
        i += 1
    correctRate = float(rate['correct']) / rate['total']
    print "在训练集中检测了", str(rate['total']), "个样本，模型在训练集中的正确率为", str(correctRate * 100), "%"
    if correctRate > accuracy:
        flag = 1
    return flag


def calcTargetFunction(alphas, dataSet):
    # 计算目标函数
    i = 0
    m, n = shape(dataSet)
    sumResult1 = 0
    alphaSum = 0
    while i < m:
        sumResult2 = 0
        j = 0
        while j < m:
            sumResult2 += alphas[i] * alphas[j] * dataSet[i, -1] * dataSet[j, -1] * kernel(dataSet[i], dataSet[j])
            j += 1
        sumResult1 += sumResult2
        alphaSum += alphas[i]
        i += 1
    return 0.5 * sumResult1 - alphaSum


def heuristicSelection(alphaBlockOne, alphaBlockTwo, EList, alphas, b, C, dataSet, diminution, alphaExceptionNumSet,
                       valueOfTargetFunction):
    # 启发式选择变量
    count = 0
    alpha1 = alphas[alphaBlockOne.num]
    diminutionTest = 0
    alphaTestMat = alphas.copy()
    for alpha in alphas:
        if count == alphaBlockOne.num:
            count += 1
            continue
        alphaBlockNewTestTwo = alphaClass()
        alphaBlockNewTestTwo.alpha = alpha
        alphaBlockNewTestTwo.num = count
        alphaBlockNewTestOne, alphaBlockNewTestTwo = alphaSolver(alphaBlockOne, alphaBlockNewTestTwo, EList, alphas,
                                                                 dataSet, C)
        alphaTestMat[count] = alphaBlockNewTestTwo.alpha
        alphaTestMat[alphaBlockNewTestOne.num] = alphaBlockNewTestOne.alpha
        diminutionTest = valueOfTargetFunction - calcTargetFunction(alphaTestMat, dataSet)
        if diminutionTest >= diminution:
            return alphaBlockNewTestOne, alphaBlockNewTestTwo
        count += 1
    alphaExceptionNumSet.add(alphaBlockOne.num)
    return None


def alphasTest(alphaBlockOne, alphaBlockTwo, alphas, alphaExceptionNumSet):
    """
    :type alphaExceptionNumSet: set
    """
    alphasTestMat = alphas.copy()
    alphasTestMat[alphaBlockOne.num] = alphaBlockOne.alpha
    alphasTestMat[alphaBlockTwo.num] = alphaBlockTwo.alpha
    return alphasTestMat


def SMO(dataSet, accuracy, C, iterNum, diminution):
    # 序列最小最优化算法主函数
    """
    :type acurracy: float
    :type dataSet: numpy.matrixlib.defmatrix.matrix
    """
    alphaExceptionNumSet = set()
    # 初始化变量排除集合
    m, n = shape(dataSet)
    alphas = zeros(m)
    i = 0
    EList = zeros(m)
    b = bCalc(alphas, dataSet, C)
    count = 0
    for E in EList:
        ECalc(count, dataSet, EList, alphas, b)
        count += 1
    flag = 0
    k = 0
    while (flag != 1 and k <= iterNum):
        valueOfTargetFunction = calcTargetFunction(alphas, dataSet)
        # 计算当前目标函数值
        alphaBlockOne = searchAlphaOne(alphas, dataSet, C, b, alphaExceptionNumSet)
        # 搜索变量一
        alphaBlockTwo = searchAlphaTwo(alphaBlockOne, EList, alphas, dataSet)
        # 一般法搜索变量二
        alphaBlockOne, alphaBlockTwo = alphaSolver(alphaBlockOne, alphaBlockTwo, EList, alphas, dataSet, C)
        # 求解两个变量的二次规划
        alphasTestMat = alphasTest(alphaBlockOne, alphaBlockTwo, alphas, alphaExceptionNumSet)
        # 构建实验变量集合
        if valueOfTargetFunction - calcTargetFunction(alphasTestMat, dataSet) < diminution:
            # 如果精度提升幅度小于预设最小缩小量后启用启发式选择变量法
            alphaBlockOne, alphaBlockTwo = heuristicSelection(alphaBlockOne, alphaBlockTwo, EList, alphas, b, C,
                                                              dataSet, diminution, alphaExceptionNumSet,
                                                              valueOfTargetFunction)
            # 启发式搜索变量二
            if alphaBlockOne is None:
                # 启发式搜索未达到目标，将已搜索变量一编号加入变量排除集合，重新搜索变量一
                k += 1
                print "支持向量机已迭代", k, "次，启发式搜索未匹配到合适的变量一，重新迭代"
                continue
        alphaExceptionNumSet.clear()
        # 清空排除alpha编号集合
        b = bUpdater(alphaBlockOne, alphaBlockTwo, alphas, dataSet, EList, b, C)
        # 更新阈值b
        alphasUpdate(alphaBlockOne, alphaBlockTwo, alphas)
        # 更新变量集合
        EUpdater(EList, alphas, b, dataSet, C)
        # 更新误差列表
        flag = terminateCondition(alphas, dataSet, b, C, accuracy)
        # 测试是否满足停机条件
        k += 1
        # 迭代次数加1
        print "支持向量机已迭代", k, "次"
    print "支持向量机学习完毕"
    return alphas, b


def test(alphas, b, dataSet, path):
    print "支持向量机开始测试"
    testSet = loadDataSet(path)
    rate = {"correct": 0, "total": 0}
    i = 0
    for data in testSet:
        rate['total'] += 1
        result = data[0, -1] * signFunction(data, dataSet, alphas, b)
        if result > 0:
            rate['correct'] += 1
        i += 1
    correctRate = float(rate['correct']) / rate['total']
    print "在测试集中检测了", str(rate['total']), "个样本，模型在测试集中的正确率为", str(correctRate * 100), "%"


dataSet = loadDataSet(r'data\plrx\plrx.txt')
alphas, b = SMO(dataSet, 0.9, 0.1, 500, 0.01)
test(alphas, b, dataSet, r'data\plrx\plrxTest.txt')
