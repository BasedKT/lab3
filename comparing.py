import random

import numpy as np

from linreg import sgd_handler, Methods, visualise_linear, gen_linear_reg, poly, poly_array, LinearRegression
from main import dogleg, gauss_newton, bfgs


def refresh_linreg(linreg, start_x):
    linreg.W = np.copy(start_x)
    linreg.W_points = [np.copy(linreg.W)]


def comparing(linreg_func, title, visualize_prev_methods=True):
    start_points = [np.array([float(random.randint(10, 40)), float(random.randint(10, 50))]) for _ in range(10)]
    linregs_examples = [gen_linear_reg(1, random.randint(10, 100), -5., 5., -10., 10., 1.) for _ in range(6)]
    lr = lambda *args: 0.01

    for linreg in linregs_examples:
        start_x = start_points[random.randint(0, len(start_points) - 1)]
        refresh_linreg(linreg, start_x)
        points = linreg_func(np.copy(start_x), linreg)
        visualise_linear(linreg.loss, points, title, "w_1", "w_2")

        if visualize_prev_methods:
            for method in Methods:
                refresh_linreg(linreg, start_x)
                sgd_handler(linreg, lr, method, store_points=True)
                sgd_points = linreg.W_points
                visualise_linear(linreg.loss, sgd_points, method.name, "w_1", "w_2")


def gauss_newton_comparing():
    def calc_points(start_x, linreg):
        funcs = np.array([lambda W, i=i: np.dot(linreg.T[i], W) - linreg.Y[i] for i in range(len(linreg.X))])
        return gauss_newton(start_x, funcs, store_points=True)

    comparing(calc_points, "Gauss-Newton", visualize_prev_methods=False)


def dogleg_comparing():
    def calc_points(start_x, linreg):
        return dogleg(linreg.loss, start_x, store_points=True)

    comparing(calc_points, "Dogleg", visualize_prev_methods=False)


def bfgs_comparing():
    def calc_points(start_x, linreg):
        return bfgs(linreg.loss, start_x, store_points=True)

    comparing(calc_points, "BFGS", visualize_prev_methods=False)


gauss_newton_comparing()
dogleg_comparing()
bfgs_comparing()
