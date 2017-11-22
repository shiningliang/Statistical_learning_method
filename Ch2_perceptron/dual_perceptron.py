# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

train_set = np.array([[[3, 3], 1], [[4, 3], 1], [[1, 1], -1], [[5, 2], -1]])

alpha = np.zeros(len(train_set), np.float)
b = 0
eta = 0.5
Gram = None
y = np.array(train_set[:, 1])
x = np.empty((len(train_set), 2), np.float)
for i in range(len(train_set)):
    x[i] = train_set[i][0]
history = []


def cal_gram():
    """
    calculate the Gram matrix
    :return:
    """
    g = np.empty((x.shape[0], x.shape[0]), np.float)
    for i, m in enumerate(x):
        for j, n in enumerate(x):
            g[i][j] = np.dot(m, n)
    return g


def update(i):
    """
    update parameters using stochastic gradient descent
    :param i:
    :return:
    """
    global alpha, b
    alpha[i] += eta
    b += (eta * y[i])
    history.append([np.dot(alpha * y, x), b])


def cal(i):
    """
    calculate the judge condition
    :param i:
    :return: judge condition
    """
    global alpha, b, x, y
    res = np.dot(alpha * y, Gram[i])
    res = (res + b) * y[i]
    return res


def check():
    """
    check if the hyperplane can classify the examples correctly
    :return: True/False
    """
    global alpha, b, x, y
    flag = False
    for i in range(x.shape[0]):
        if cal(i) <= 0:
            flag = True
            update(i)
    w = np.dot(alpha * y, x)
    print('w: ' + str(w) + '\tb: ' + str(b))
    if not flag:
        return False
    return True


if __name__ == '__main__':
    Gram = cal_gram()
    for i in range(1000):
        print('Iteration ', i + 1)
        if not check(): break

    # draw an animation to show how it works, the data comes from history
    # first set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
    line, = ax.plot([], [], 'g', lw=2)
    label = ax.text([], [], '')


    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        x, y, x_, y_ = [], [], [], []
        for p in train_set:
            if p[1] > 0:
                x.append(p[0][0])
                y.append(p[0][1])
            else:
                x_.append(p[0][0])
                y_.append(p[0][1])

        plt.plot(x, y, 'bo', x_, y_, 'rx')
        plt.axis([-6, 6, -6, 6])
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('dual Perceptron')
        return line, label


    # animation function.  this is called sequentially
    def animate(i):
        global history, ax, line, label

        w = history[i][0]
        b = history[i][1]
        if w[1] == 0: return line, label
        x1 = -7.0
        y1 = -(b + w[0] * x1) / w[1]
        x2 = 7.0
        y2 = -(b + w[0] * x2) / w[1]
        line.set_data([x1, x2], [y1, y2])
        x1 = 0.0
        y1 = -(b + w[0] * x1) / w[1]
        label.set_text(str(history[i][0]) + ' ' + str(b))
        label.set_position([x1, y1])
        return line, label


    # call the animator.  blit=true means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(history), interval=1000, repeat=True,
                                   blit=True)
    plt.show()
