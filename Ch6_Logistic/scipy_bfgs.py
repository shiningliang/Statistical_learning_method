from scipy.optimize import fmin_bfgs
import numpy as np

def func(x):
    return 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2

x = [-10, -10]
xopt = fmin_bfgs(func, x)
print(xopt)
