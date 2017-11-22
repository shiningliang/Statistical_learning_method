from numpy import *
import matplotlib.pyplot as plt


def loadSimData():
    """
    加载简单数据集，二维平面上的5个点
    :return:
    """
    datMat = matrix([[1.0, 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])

    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def plotData(datMat, classLabels):
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []

    for i in range(len(classLabels)):
        if classLabels[i] == 1.0:
            xcord1.append(datMat[i, 0]), ycord1.append(datMat[i, 1])
        else:
            xcord0.append(datMat[i, 0]), ycord0.append(datMat[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord0, ycord0, marker='s', s=90)
    ax.scatter(xcord1, ycord1, marker='o', s=50, c='red')
    plt.title('decision stump test data')
    plt.show()


def stumpClassify(dataMatrix, dimen, threshVal, threshInsq):
    """
    用只有一层的树桩决策树对数据进行分类
    :param dataMatrix: 数据
    :param dimen: 特征维度
    :param threshVal: 阈值
    :param threshInsq: 大于或小于
    :return: 分类结果
    """
    retArray = ones((dataMatrix.shape[0], 1))
    if threshInsq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    """
    构建决策树（树桩）
    :param dataArr: 数据
    :param classLabels: 标签
    :param D: 数据权重
    :return: 最优决策树,最小的错误率加权和,最优预测结果
    """
    dataMatrix = mat(dataArr)  # matrix必须是二维，numpy可以是多维
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)  # m 数据个数 n 特征数
    numSteps = 10.0  # 在特征的所有可能值遍历的步长
    bestStump = {}  # 用于存储给定权重向量0时所得到的最佳单层决策树的相关信息
    bestClassEnt = mat(zeros((m, 1)))
    minError = inf  # 首先将minError初始化为正无穷大
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            # lt ：小于，lte，le：小于等于
            # gt：大于，gte，ge：大于等于
            # eq：等于  ne,neq：不等于
            for inequal in ['lt', 'gt']:  # 遍历大于和小于
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)

                # 计算误差,初始化错误矩阵为1，如果判断正确则设置为0
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                # 计算加权错误概率
                weightedError = D.T * errArr

                # 更新bestStump中保存的最佳单层决策树
                if weightedError < minError:
                    minError = weightedError
                    bestClassEnt = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    return bestStump, minError, bestClassEnt


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    基于单层决策树ada训练
    :param dataArr: 数据
    :param classLabels: 标签
    :param numIt: 迭代次数
    :return: 一系列弱分类器及其权重,样本分类结果
    """
    weakClassArr = []
    m = dataArr.shape[0]
    D = mat(ones((m, 1)) / m)  # 数据权值初始化
    aggClassEst = mat(zeros((m, 1)))
    # 迭代
    for i in range(numIt):
        # 调用单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))  # max(error,1e-16)))用于确保没有错误时，不会发生溢出
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # 保存决策树
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)  # 每个样本对应的指数,当预测值等于y时,为-alpha,否则为alpha
        D = multiply(D, exp(expon))  # 计算下一个迭代的D向量
        D = D / D.sum()  # 归一化

        # 计算所有分类器的误差,如果为0则终止训练
        aggClassEst += alpha * classEst
        print("aggClassEst:", aggClassEst.T)

        # aggClassEst每个元素的符号代表分类结果,如果与y不等则表示错误
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error:", errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr


def adaClassify(datToClass, classifierArr):
    """
    AdaBoost分类函数
    :param datToClass: 待分类样例
    :param classifierArr: 多个弱分类器
    :return:
    """
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return sign(aggClassEst)


# main函数
if __name__ == "__main__":
    # 加载数据集
    datMat, classLabels = loadSimData()
    # plotData(datMat, classLabels)

    # 基于单层决策树的Adaboost训练过程
    classifierArray = adaBoostTrainDS(datMat, classLabels, 30)

    # 测试AdaBoost分类函数
    print("\n\n[[5,5],[0,0]]:\n", adaClassify([[5, 5], [0, 0]], classifierArray))
