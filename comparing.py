import random

import numpy as np

from linreg import sgd_handler, Methods, visualise_linear, gen_linear_reg
from main import dogleg


def refresh_linreg(linreg, start_x):
    linreg.W = np.copy(start_x)
    linreg.W_points = [np.copy(linreg.W)]


def comparing():
    start_x = np.array([30., 30.])
    linregs_examples = [gen_linear_reg(1, random.randint(10, 50), -5., 5., -10., 10., 1.) for _ in range(6)]
    lr = lambda *args: 0.01

    for linreg in linregs_examples:
        refresh_linreg(linreg, start_x)
        dogleg_points = dogleg(linreg.loss, linreg.W, store_points=True)
        visualise_linear(linreg.loss, dogleg_points, "Dogleg", "w_1", "w_2")
        for method in Methods:
            refresh_linreg(linreg, start_x)
            sgd_handler(linreg, lr, method, store_points=True)
            sgd_points = linreg.W_points
            visualise_linear(linreg.loss, sgd_points, method.name, "w_1", "w_2")



comparing()
