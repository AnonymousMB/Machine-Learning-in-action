# coding=utf-8
from numpy import *
from matplotlib import pyplot as plt


def loadDataSet(path):
    # 数据集读取函数
    fr = open(path, 'r')
    dataMat = []
    labelMat = []
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append(lineArr[0:-1])
        labelMat.append(lineArr[-1])
    dataMat = mat(dataMat, dtype=float64)
    labelMat = mat(labelMat, dtype=float64)
    return dataMat, labelMat


class Data(object):
    def __init__(self, attr, label, alpha, num):
        self.attr = attr
        # 输入特征
        self.label = label
        # 类别标签
        self.alpha = alpha
        # 自由变量
        self.num = num
        # 样本序号
        self.E = None
        # 该样本经过输出函数计算后与实际类别的误差（g(x)-label）
        self.oldAlpha = None
        # 缓存的旧自由变量


def kernel(xi, xj):
    # 核函数
    """
    :type xj: matrix
    :type xi: matrix
    """
    '''vec=xi-xj
    L2power=float(sum(multiply(vec,vec)))
    theta=sqrt(2)
    output=exp(-L2power/(2*(theta**2)))'''
    output = xi * xj.transpose()
    # 输入特征作内积
    return output


def transformDataMatToDataSet(dataMat, labelMat, alphas):
    dataSet = []
    # 初始化dataSet
    rowNum = 0
    # 初始化行数
    m, n = shape(dataMat)
    for rowNum in range(m):
        dataRow = dataMat[rowNum, :]
        data = Data(dataRow, labelMat[0, rowNum], alphas[rowNum], rowNum)
        # 构建一个data实例
        dataSet.append(data)
        rowNum += 1
    dataSet = mat(dataSet)[0]
    # 将dataSet转变成矩阵
    return dataSet


def hypothesis(dataX, dataSet, b):
    # 输出函数g(x)
    """
    :type dataSet: matrix
    :type dataX: Data
    """
    j = 0
    sumResult = 0
    m, n = shape(dataSet)
    # 得到样本集合中样本的数目m
    for j in range(n):
        data = dataSet[0, j]
        # 从样本集合中取出样本
        sumResult += data.alpha * data.label * kernel(data.attr, dataX.attr)
        # 计算g(x)第一项
    result = sumResult + b
    return result


def ECalc(dataX, dataSet, b):
    # 计算对应样本dataX的误差偏量
    """
    :type dataX: Data
    """
    result = hypothesis(dataX, dataSet, b)
    # 样本dataX经输出函数输出的值
    dataX.E = result - dataX.label
    # 计算得到的误差，并将其记录在样本实例中
    return dataX.E


def solveAlpha(data1, data2, dataSet, b, C):
    # 求解二次规划
    """
    :type data2: Data
    :type data1: Data
    """

    def alphaCutter(alpha2NewUnc, data1, data2, C):
        # 求解二次规划辅助函数：alpha2剪辑函数
        """
        :type data1: Data
        :type data2: Data
        :type alpha2NewUnc: float
        """
        print "开始求解两个变量"
        L = 0
        H = 0
        # 初始化边界约束条件L和H
        if data1.label != data2.label:
            # 如果两个样本的类别不同
            L = max(data2.oldAlpha - data1.oldAlpha, 0)
            H = min(C, C + data2.oldAlpha - data1.oldAlpha)
        else:
            # 如果两个样本的类别相同
            L = max(data2.oldAlpha + data1.oldAlpha - C, 0)
            H = min(C, data2.oldAlpha + data1.oldAlpha)
        alpha = 0
        # 初始化新alpha
        if alpha2NewUnc > H:
            alpha = H
        elif alpha2NewUnc >= L and alpha2NewUnc <= H:
            alpha = alpha2NewUnc
        elif alpha2NewUnc < L:
            alpha = L
        return alpha

    eta = kernel(data1.attr, data1.attr) + kernel(data2.attr, data2.attr) - 2 * kernel(data1.attr, data2.attr)
    # 计算分母eta=K11+K12-2K12
    data1.oldAlpha = data1.alpha
    data2.oldAlpha = data2.alpha
    # 将样本data1和data2未被更新的alpha值存入缓存
    alpha2NewUnc = data2.oldAlpha + float(data2.label * (data1.E - data2.E)) / eta
    alpha2NewUnc = alpha2NewUnc[0, 0]
    # 更新无约束情况下样本data2的alpha值
    data2.alpha = alphaCutter(alpha2NewUnc, data1, data2, C)
    # 将剪辑后的alpha2值存入样本data2
    data1.alpha = float(data1.oldAlpha + data1.label * data2.label * (data2.oldAlpha - data2.alpha))
    # 更新样本data1对应的alpha
    print "求解完毕，返还更新alpha后的样本"
    return data1, data2


def selectAlpha1(dataSet, C, b, abandomAlphaNum, tol):
    # 搜索第一个变量alpha1
    """
    :type dataSet: matrix
    """
    weight = 0
    # 初始化违反KKT条件的程度
    data1 = Data
    # 初始化选中的alpha1对应的样本
    m, n = shape(dataSet)
    count = 0
    print "开始搜索变量1"
    for count in range(n):
        # 在整个训练集循环寻找第一个变量alpha
        if count in abandomAlphaNum:
            # 如果发现当前搜索到的alpha是被抛弃的，则直接搜索下一个alpha
            count += 1
            continue
        data = dataSet[0, count]
        condition = data.label * hypothesis(data, dataSet, b)
        # 用y*g(x)计算条件值(实际上就是样本距离超平面的函数间隔)
        if data.alpha ==0 and condition < 1 - tol:
            # 违反KKT并且超过最大容错量
            if weight < abs(condition - 1):
                # 违反程度大于已知最大程度
                weight = abs(condition - 1)
                # 存入违反程度
                data1 = data
        elif data.alpha ==C and condition > 1 + tol:
            # 违反KKT并且超过最大容错量
            if weight < abs(condition - 1):
                # 违反程度大于已知最大程度
                weight = abs(condition - 1)
                # 存入违反程度
                data1 = data
        elif data.alpha>0 and data.alpha <C and abs(condition-1)>tol:
            # 违反KKT并且超过最大容错量
            if weight < abs(condition - 1):
                # 违反程度大于已知最大程度
                weight = abs(condition - 1)
                # 存入违反程度
                data1 = data
        count += 1
    print "变量1搜索完毕,值为", data1.alpha
    return data1
    # 返还搜索到的合适样本


def selectAlpha2(data1, dataSet, abandomAlpha2Num):
    # 第二个变量选择
    print "开始搜索用于优化的第二个变量"
    deviation = 0
    # 初始化误差差值为0
    data2 = None
    # 初始化第二个搜索到的样本data2
    m, n = shape(dataSet)
    count = 0
    for count in range(n):
        if count == data1.num or count in abandomAlpha2Num:
            # 如果当前检索的样本与data1是同一个样本或者是被抛弃的啊alpha2，则直接跳过，搜索下一个样本
            count += 1
            continue
        data = dataSet[0, count]
        if deviation < abs(data.E - data1.E):
            # 如果找到更大的|E1-E2|
            deviation = abs(data.E - data1.E)
            data2 = data
        count += 1
    if data2 is not None:
        print "变量2搜索完毕，值为", data2.alpha
    return data2
    # 返还搜索到的data2


def updateEandb(data1, data2, dataSet, b, C):
    # 更新E和b
    """
    :type data2: Data
    :type data1: Data
    """
    bNew = 0
    # 初始化新的b值
    b1 = -data1.E - data1.label * kernel(data1.attr, data1.attr) * (data1.alpha - data1.oldAlpha) \
         - data2.label * kernel(data2.attr, data1.attr) * (data2.alpha - data2.oldAlpha) + b
    # b1=-E1-y1K11(a1new-a1new)-y2K21(a2new-a2old)+bold
    b2 = -data2.E - data1.label * kernel(data1.attr, data2.attr) * (data1.alpha - data1.oldAlpha) \
         - data2.label * kernel(data2.attr, data2.attr) * (data2.alpha - data2.oldAlpha) + b
    # b2=-E2-y1K12(a1new-a1old)-y2K22(a2new-a2old)+bold
    if data1.alpha > 0 and data1.alpha < C and data2.alpha > 0 and data2.alpha < C:
        # 如果选取的两个样本都是支持向量
        bNew = float(b1)
    else:
        # 如果不是
        bNew = float(b1 + b2) / 2
    # b更新完毕
    print "阈值b更新完毕，值为", bNew
    count = 0
    m, n = shape(dataSet)
    sumResult1 = 0
    sumResult2 = 0
    i = 0
    for count in range(n):
        data = dataSet[0, count]
        if data.alpha > 0 and data.alpha < C:
            sumResult1 += data.label * data.alpha * kernel(data1.attr, data.attr)
            sumResult2 += data.label * data.alpha * kernel(data2.attr, data.attr)
        count += 1
    data1.E =float( sumResult1 + bNew - data1.label)
    data2.E = float(sumResult2 + bNew - data2.label)
    # E值更新完毕
    print "E值更新完毕，分别为", data1.E, data2.E
    return bNew
    # 返还更新后的阈值b


def signFunc(dataX, dataSet, b):
    # 符号函数
    result = hypothesis(dataX, dataSet, b)
    if result > 0:
        return 1
    else:
        return -1


def terminate(dataSet, b, C, tol):
    m, n = shape(dataSet)
    rate = {'correct': 0, 'total': 0}
    i = 0
    flag = 1
    for count in range(n):
        data = dataSet[0, count]
        condition = data.label * hypothesis(data, dataSet, b)
        # 用y*g(x)计算条件值
        if data.alpha == 0 and condition < 1 - tol:
            # 违反KKT并且超过最大容错量
            flag = 0
            break
        elif data.alpha == C and condition > 1 + tol:
            # 违反KKT并且超过最大容错量
            flag = 0
            break
        elif data.alpha>0 and data.alpha <C and abs(condition-1)>tol:
            # 违反KKT并且超过最大容错量
            flag=0
            break
        count += 1
    for i in range(n):
        data = dataSet[0, i]
        rate['total'] += 1
        result = signFunc(data, dataSet, b)
        if result == data.label:
            rate['correct'] += 1
        i += 1
    correctRate = float(rate['correct']) / rate['total']
    print "正确率为", correctRate * 100
    return flag


def SMO(dataMat, labelMat, C, tol, iterNum, diminution):
    # SMO主体函数
    """
    :rtype: matrix
    :param diminution: 目标函数最小下降量
    :param iterNum: 最大迭代次数
    :type C: float
    :param C: 罚项
    :type tol: float
    :param tol: 预设容错度（最小允许误差）
    """
    b = 0
    # 初始化阈值b
    m, n = shape(dataMat)
    alphas = zeros(m)
    # 初始化所有alpha为0
    dataSet = transformDataMatToDataSet(dataMat, labelMat, alphas)
    # 整合数据内容为dataSet矩阵,dataSet矩阵的维数为1*n
    k = 0
    # 已迭代次数
    flag = 0
    # 停机条件判断标签
    abandomAlpha1Num = set()
    abandomAlpha2Num = set()
    # 初始化空的已排除alpha编号集合
    for i in range(m):
        # 这里的m用的是dataMat的
        data = dataSet[0, i]
        ECalc(data, dataSet, b)
    # 初始化所有样本对应的误差E
    while (flag == 0 and k < iterNum):
        # 如果满足停机条件或者迭代次数达到最大值则结束循环，否则继续
        data1 = selectAlpha1(dataSet, C, b, abandomAlpha1Num, tol)
        # 搜索变量1
        data2 = selectAlpha2(data1, dataSet, abandomAlpha2Num)
        # 搜索变量2
        data1, data2 = solveAlpha(data1, data2, dataSet, b, C)
        # 求解二次规划
        while abs(data2.oldAlpha - data2.alpha) < diminution:
            # 如果选中的alpha2无法造成足够的下降并且还有未检索的alpha2
            print "选中的alpha2无法造成足够下降，重新寻找合适的alpha2"
            abandomAlpha2Num.add(data2.num)
            data2 = selectAlpha2(data1, dataSet, abandomAlpha2Num)
            if data2 is None:
                break
            data1, data2 = solveAlpha(data1, data2, dataSet, b, C)
        if data2 is None:
            # 如果所有的alpha2都不满足条件
            print "检索所有的alpha2发现效果不佳，选中的alpha1无法造成足够下降，重新寻找合适的alpha1"
            abandomAlpha1Num.add(data1.num)
            abandomAlpha2Num.clear()
            k += 1
            print "并且迭代次数+1，完成第", k, "次迭代"
            continue
        abandomAlpha2Num.clear()
        abandomAlpha1Num.clear()
        b = updateEandb(data1, data2, dataSet, b, C)
        # 更新E和b
        flag = terminate(dataSet, b, C, tol)
        # 判断是否满足停机条件
        k += 1
        countT(dataMat, labelMat, dataSet, alphas, b, C)
        # 计算斜率
        print "第", k, "次迭代完成\n"

    i = 0
    for i in range(m):
        alphas[i] = dataSet[0, i].alpha
        i += 1
    return alphas, b


def draw(dataMat, labelMat, dataSet, alphas, b, C):
    figure = plt.figure()
    ax = figure.add_subplot(111)
    m, n = shape(dataSet)
    i = 0
    for i in range(n):
        data = dataSet[0, i]
        if data.label == 1:
            ax.scatter(data.attr[0, 0], data.attr[0, 1], c="b")
        else:
            ax.scatter(data.attr[0, 0], data.attr[0, 1], c="g")
    for i in range(n):
        data = dataSet[0, i]
        if data.alpha > 0 and data.alpha < C:
            ax.scatter(data.attr[0, 0], data.attr[0, 1], c="r")
    w = multiply(alphas, labelMat) * dataMat
    X = arange(-2, 10, 0.1)
    Y = (-b - w[0, 0] * X) / w[0, 1]
    try:
        m, n = shape(Y)
        Y = array(Y)[0]
    except:
        Y = array(Y)
    ax.plot(X, Y)
    plt.show()


def countT(dataMat, labelMat, dataSet, alphas, b, C):
    # 计算斜率
    i = 0
    m, n = shape(dataMat)
    for i in range(m):
        alphas[i] = dataSet[0, i].alpha
        i += 1
    w = multiply(alphas, labelMat) * dataMat
    X = arange(-2, 10, 0.1)
    Y = (-b - w[0, 0] * X) / w[0, 1]
    try:
        m, n = shape(Y)
        Y = array(Y)[0]
    except:
        Y = array(Y)
    Y1 = Y[0]
    Y2 = Y[1]
    a2 = float(b * Y1 - b * X[0]) / (X[0] * Y2 - X[1] * Y1)
    a1 = float(-b - a2 * X[1]) / X[0]
    t = -a1 / a2
    print "斜率为", t


dataMat, labelMat = loadDataSet(r'data\mlia\Ch06\testSet.txt')
alphas, b = SMO(dataMat, labelMat, 0.6, 0, 20, 0.00001)
dataSet = transformDataMatToDataSet(dataMat, labelMat, alphas)
draw(dataMat, labelMat, dataSet, alphas, b, 5)
