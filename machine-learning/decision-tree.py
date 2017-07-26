# coding=utf-8
from numpy import *
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
import pandas as pd
import copy


def loadDataSet():
    dataSet = pd.read_csv(r'data\watermalon\2.0.txt', encoding='gb2312')
    return dataSet


def calcEntropy(dataSet, nodeName):  # 接收数据集和节点名作为参数
    goodSet = dataSet[dataSet[u'好瓜'].str.contains(u'是')]
    badSet = dataSet[dataSet[u'好瓜'].str.contains(u'否')]  # 将好瓜与坏瓜分组
    goodCounts = {}
    badCounts = {}
    totalCounts = {}
    ent = {}
    k = 0
    for column in dataSet:
        if k == 0:  # 跳过编码列
            k = k + 1
            continue
        totalCounts[column] = dataSet[column].value_counts()  # 对整个样本各属性的属性值计数
        goodCounts[column] = goodSet[column].value_counts()  # 对好瓜样本的属性值计数
        badCounts[column] = badSet[column].value_counts()  # 对坏瓜样本的属性值计数
        ent[column] = {}
        group = dataSet.groupby(column)  # 对当前样本集按照属性分组
        for name, content in group:
            if nodeName == name and len(ent) != 0:  # 如果是当前节点所代表的列则直接跳过
                continue
            goodNum = 0
            badNum = 0
            totalNum = 0
            if nodeName == column:
                totalNum = totalCounts[column].sum()
            else:
                totalNum = totalCounts[column][name]
            if name in goodCounts[column]:
                goodNum = float(goodCounts[column][name])
            else:
                goodNum = totalNum
            if name in badCounts[column]:
                badNum = float(badCounts[column][name])
            else:
                badNum = totalNum
            entropy = 0 - (float(goodNum / totalNum * math.log(goodNum / totalNum, 2)) + float(
                badNum / totalNum * math.log(badNum / totalNum, 2)))
            ent[column][name] = entropy
            # 这一段有比我更好的写法，我的很臃肿
    return ent


def calcGain(dataSet, nodeName, attributeSet):
    ent = calcEntropy(dataSet, nodeName)
    length = len(dataSet)
    gain = {}
    for column in attributeSet:
        sumResult = 0
        group = dataSet.groupby(column)
        for name, content in group:
            sumResult = sumResult + float(len(content)) / length * ent[column][name]
        if isinstance(nodeName, unicode):  # 如果是根节点
            gain[column] = ent[nodeName][u'是'] + ent[nodeName][u'否'] - sumResult
        elif isinstance(nodeName, tuple):  # 如果不是根节点
            gain[column] = ent[nodeName[0]][nodeName[1]] - sumResult
    return gain


def chooseDataSet(dataSet, nodeName, attributeSet):
    gain = calcGain(dataSet, nodeName, attributeSet)
    bestAttribute = None
    bestGainValue = 0
    for attribute, gainValue in gain.iteritems():
        if attribute in attributeSet and gainValue > bestGainValue:  # 选择使信息增益最大的属性
            bestGainValue = gainValue
            bestAttribute = attribute
    return bestAttribute


def createTree():
    dataSet = loadDataSet()  # 读取数据
    root = {}  # 初始化根节点
    attributeSet = set()
    nodeNameList = []
    dataSet = loadDataSet()
    data = dataSet.ix[:, 1:-2]
    for column in data:
        attributeSet.add(column)
    for column in dataSet:
        nodeNameList.append(column)
    nodeName = nodeNameList[-1]
    root['root'] = treeGenerate(dataSet, attributeSet, nodeName)
    return root


def treeGenerate(dataSet, attributeSet, nodeName):
    """
    :type attributeSet: tuple ,dataSet: DataFrame
    """
    node = {}
    nameSet = set()
    if dataSet is None:
        return
    # 样本集为空，直接返还
    i = 0
    name = None
    resultCount = {}
    group = dataSet.groupby(dataSet.ix[:, -1])
    for name, content in group:
        i = i + 1
        resultCount[name] = len(content)
    if i == 1:
        node[dataSet.ix[:, -1].name] = name
        return node
    # 如果dataSet中样本全部属于同一类别，那么返回叶节点
    if len(attributeSet) == 1 or len(attributeSet) == 0:
        node[dataSet.ix[:, -1].name] = max(resultCount)
        return node
    # 如果属性集合中只剩下编码列或为空，返回叶节点
    bestAttribute = chooseDataSet(dataSet, nodeName, attributeSet)
    if bestAttribute is None:
        node[dataSet.ix[:, -1].name] = max(resultCount)
        return node
    # 获取最佳划分属性
    group = dataSet.groupby(dataSet[bestAttribute])
    # 按照属性分组
    attributeSet = copy.copy(attributeSet)
    attributeSet.remove(bestAttribute)
    # 从属性集合中删除划分过的属性
    for name, content in group:
        node[(bestAttribute, name)] = treeGenerate(content, attributeSet, (bestAttribute, name))
    # 根据划分创建分支节点
    return node


def getNumLeafs(tree):
    numLeafs = 0
    for key in tree.keys():
        secondDict = tree[key]
        if isinstance(secondDict, dict):
            numLeafs += getNumLeafs(secondDict)
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(tree):
    maxDepth = 0
    for key in tree.keys():
        secondDict = tree[key]
        if isinstance(secondDict, dict):
            thisDepth = 1 + getTreeDepth(secondDict)
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotNode(nodeName, finalLoc, startLoc, nodeType):
    createPlot.ax1.annotate(nodeName, xy=startLoc, xycoords='axes fraction', xytext=finalLoc, textcoords='axes fraction'
                            , bbox=nodeType, arrowprops=arrow_args)  # startLoc 箭头起点 ，finalLoc 箭头指向的终点


def createPlot(tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotTree.totalW = float(getNumLeafs(tree))
    plotTree.totalD = float(getTreeDepth(tree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1
    plotTree(tree, (0.5, 1.0), '')
    plt.show()


def plotTree(tree, startLoc, nodeName):
    numLeafs = float(getNumLeafs(tree))
    # 得到当前节点要占的宽度
    depth = float(getTreeDepth(tree))
    # 得到当前节点要占的高度
    finalLoc = (plotTree.xOff + (1.0 + numLeafs) / 2.0 / plotTree.totalW, plotTree.yOff)
    # 计算当前节点的位置
    plotNode(nodeName, finalLoc, startLoc, decisionNode)
    # 绘制当前节点
    plotTree.yOff -= 1.0 / plotTree.totalD
    # 减少y偏移
    for node in tree.keys():  # 检索当前节点的分支
        if isinstance(tree[node], dict):
            # 如果是子节点
            text = ''
            if isinstance(node, tuple):  # 判断是否是根节点
                text=node[0].encode('utf-8')+':'+node[1].encode('utf-8')
                #将原本是unicode类型的节点键编码为‘utf-8’格式的str类以方便显示图表
            else:
                text = 'root'
            plotTree(tree[node], finalLoc, text)
        else:
            # 如果是叶节点
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            if tree[node] == u'是':
                tree[node] = u'好瓜'
            else:
                tree[node] = u'坏瓜'
            plotNode(tree[node], (plotTree.xOff, plotTree.yOff), finalLoc, leafNode)
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


tree = createTree()
print tree
decisionNode = dict(boxstyle="square", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

createPlot(tree)
