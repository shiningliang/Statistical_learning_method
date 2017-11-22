import numpy as np
import matplotlib.pyplot as plt

# 函数表达式fun
fun = lambda x: 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2

# 梯度向量 gfun
gfun = lambda x: np.array([400 * x[0] * (x[0] ** 2 - x[1]) + 2 * (x[0] - 1), -200 * (x[0] ** 2 - x[1])])

# 海森矩阵 hess
hess = lambda x: np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]])


def bfgs(fun, gfun, hess, x0):
    """
    Minimize a function func using the BFGS algorithm.
    :param fun: f(x) Function to minimise.
    :param gfun: gradient of f(x)
    :param hess: hessian matrix
    :param x0: initial point
    :return: optimal point, optimal f(x), iter nums
    """
    maxk = 1e5
    beta = 0.55
    sigma = 0.4
    epsilon = 1e-5  # 阈值
    k = 0
    n = np.shape(x0)[0]
    # 海森矩阵可以初始化为单位矩阵
    # B_k = np.linalg.inv(hess(x0))
    B_k = np.eye(n)

    while k < maxk:
        g_k = gfun(x0)
        if np.linalg.norm(g_k) < epsilon:
            break
        p_k = -1.0 * np.linalg.solve(B_k, g_k)
        m = 0
        m_k = 0
        while m < 20:  # 用Armijo搜索求步长
            if fun(x0 + beta ** m * p_k) < fun(x0) + sigma * beta ** m * np.dot(g_k, p_k):
                m_k = m
                break
            m += 1

        # BFGS校正
        lamda = beta ** m_k
        x = x0 + lamda * p_k
        s_k = x - x0
        y_k = gfun(x) - g_k

        if np.dot(s_k, y_k) > 0:
            Bs = np.dot(B_k, s_k)
            ys = np.dot(y_k, s_k)
            sBs = np.dot(np.dot(s_k, B_k), s_k)

            B_k = B_k - 1.0 * Bs.reshape((n, 1)) * Bs / sBs + 1.0 * y_k.reshape((n, 1)) * y_k / ys

        k += 1
        x0 = x

    return x0, fun(x0), k  # 分别是最优点坐标，最优值，迭代次数


n = 50
x = y = np.linspace(-10, 10, n)  # 在指定的间隔内返回均匀间隔的数字
xx, yy = np.meshgrid(x, y)  # x为矩阵xx的行向量，y为矩阵yy的列向量 xx和yy构成了一个坐标矩阵，也是一个网格
data = [[xx[i][j], yy[i][j]] for i in range(n) for j in range(n)]  # 在网格上逐行取点
iters = []
for i in range(np.shape(data)[0]):
    rt = bfgs(fun, gfun, hess, np.array(data[i]))
    if rt[2] <= 200:
        iters.append(rt[2])
    if i % 100 == 0:
        print(i, rt[0], rt[1], rt[2])

plt.hist(iters, bins=50)
plt.title(u'BFGS iter nums distribution', {'fontname': 'Monaco', 'fontsize': 14})
plt.xlabel(u'iter nums', {'fontname': 'Monaco', 'fontsize': 14})
plt.ylabel(u'freq distribution', {'fontname': 'Monaco', 'fontsize': 14})
plt.show()