import random
import tracemalloc
from time import time

import numpy as np
from matplotlib import pyplot as plt

from excel import ExcellSaver
from linreg import sgd_handler, Methods, visualise_approximation, gen_linear_reg
from main import dogleg, gauss_newton, bfgs, lbfgs


class Metrics:
    def __init__(self, mem, steps, time):
        self.mem = mem
        self.points = steps
        self.time = time


def visualise(f, points, title, x_label="x", y_label="y"):
    values = np.transpose(points)
    X = np.linspace(min(values[0]) - 10, max(values[0]) + 10, 100)
    Y = np.linspace(min(values[1]) - 10, max(values[1]) + 10, 100)
    Z = [[f(np.array([X[i], Y[j]])) for i in range(len(X))] for j in range(len(Y))]
    plt.contour(X, Y, Z, 30)

    plt.plot(values[0], values[1], marker='.')
    plt.plot(values[0][0], values[1][0], 'og')
    plt.plot(values[0][-1], values[1][-1], 'or')
    plt.title(title)
    plt.legend(['Linear Regression', 'Start point', 'End point'])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def get_metrics(func):
    start = time()
    tracemalloc.start()
    steps = func()
    mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end = time()

    return Metrics(mem, steps, end - start)


def refresh_linreg(linreg, start_x):
    linreg.W = np.copy(start_x)
    linreg.W_points = [np.copy(linreg.W)]


def linreg_comparing(linreg_func, title, visualize_prev_methods=True):
    global excel_saver

    start_points = [np.array([float(random.randint(5, 30)), float(random.randint(10, 50))]) for _ in range(10)]
    linregs_examples = [gen_linear_reg(1, random.randint(20, 100), -5., 5., -10., 10., 10.) for _ in range(6)]
    lr = lambda *args: 0.01

    for linreg in linregs_examples:
        start_x = start_points[random.randint(0, len(start_points) - 1)]
        refresh_linreg(linreg, start_x)
        metrics = get_metrics(lambda: linreg_func(np.copy(start_x), linreg))

        linreg.W = metrics.points[-1]
        linreg.W_points = metrics.points
        excel_saver.add_row([title, len(metrics.points), linreg.loss(metrics.points[-1]), metrics.mem, metrics.time])
        visualise(linreg.loss, linreg.W_points, title, "w_1", "w_2")
        visualise_approximation(linreg, title)

        if visualize_prev_methods:
            for method in Methods:
                def sgd_caller():
                    sgd_handler(linreg, lr, method, store_points=True)
                    return linreg.W_points

                refresh_linreg(linreg, start_x)
                metrics = get_metrics(sgd_caller)

                excel_saver.add_row([method, len(metrics.points), linreg.loss(metrics.points[-1]), metrics.mem,
                                     metrics.time])
                visualise(linreg.loss, linreg.W_points, method.name.replace("Methods.", ""), "w_1", "w_2")
                visualise_approximation(linreg, method.name.replace("Methods.", ""))


def gauss_newton_vs_prev():
    global excel_saver

    def calc_points(start_x, linreg):
        funcs = np.array([lambda W, i=i: np.dot(linreg.T[i], W) - linreg.Y[i] for i in range(len(linreg.X))])
        return gauss_newton(start_x, funcs, store_points=True)

    excel_saver.add_new_sheet(["method", "count steps", "loss", "memory", "time"], "Gauss-Newton vs Prev")
    linreg_comparing(calc_points, "Gauss-Newton", visualize_prev_methods=True)


def dogleg_vs_prev():
    global excel_saver

    def calc_points(start_x, linreg):
        return dogleg(linreg.loss, start_x, store_points=True)

    excel_saver.add_new_sheet(["method", "count steps", "loss", "memory", "time"], "Dogleg vs Prev")
    linreg_comparing(calc_points, "Dogleg", visualize_prev_methods=True)


def bfgs_vs_prev():
    global excel_saver

    def calc_points(start_x, linreg):
        return bfgs(linreg.loss, start_x, store_points=True)

    excel_saver.add_new_sheet(["method", "count steps", "loss", "memory", "time"], "BFGS vs Prev")
    linreg_comparing(calc_points, "BFGS", visualize_prev_methods=True)


def lbfgs_vs_prev():
    global excel_saver

    def calc_points(start_x, linreg):
        return lbfgs(linreg.loss, start_x, store_points=True)

    excel_saver.add_new_sheet(["method", "count steps", "loss", "memory", "time"], "LBFGS vs Prev")
    linreg_comparing(calc_points, "LBFGS", visualize_prev_methods=False)


def comparing_between(funcs, methods, titles, is_array_funcs=False):
    global excel_saver

    start_points = [np.array([float(random.randint(5, 30)), float(random.randint(10, 50))]) for _ in range(10)]
    for f in funcs:
        start_x = start_points[random.randint(0, len(start_points) - 1)]
        for i in range(len(methods)):
            metrics = get_metrics(lambda: methods[i](f, np.copy(start_x)))
            real_func = lambda x: sum([f[i](x) ** 2 for i in range(len(f))]) if is_array_funcs else f
            excel_saver.add_row(
                [titles[i], len(metrics.points), real_func(metrics.points[-1]), metrics.mem, metrics.time])
            visualise(real_func, metrics.points, titles[i])


def gauss_newton_vs_dogleg():
    global excel_saver

    excel_saver.add_new_sheet(["method", "count steps", "function value", "memory", "time"], "Gauss Newton vs Dogleg")
    comparing_between(
        [
            [lambda x: x[0] - 5, lambda x: x[1] + 6],
            [lambda x: 10 * (x[0] - x[1]) + 9, lambda x: x[0] * (x[1] - 12) + x[1]]
        ],
        [
            lambda funcs, start_x: gauss_newton(start_x, funcs, store_points=True),
            lambda funcs, start_x: dogleg(lambda x: sum([funcs[i](x) ** 2 for i in range(len(funcs))]), start_x,
                                          store_points=True)
        ],
        ["Gauss Newton", "Dogleg"], is_array_funcs=True
    )


def gauss_newton_vs_bfgs():
    excel_saver.add_new_sheet(["method", "count steps", "function value", "memory", "time"], "Gauss Newton vs BFGS")
    comparing_between(
        [

        ],
        [
            lambda funcs, start_x: gauss_newton(start_x, funcs, store_points=True),
            lambda funcs, start_x: bfgs(lambda x: sum([funcs[i](x) ** 2] for i in range(len(funcs))), start_x,
                                        store_points=True)
        ],
        ["Gauss Newton", "BFGS"]
    )


def dogleg_vs_bfgs():
    excel_saver.add_new_sheet(["method", "count steps", "function value", "memory", "time"], "Gauss Newton vs BFGS")
    comparing_between(
        [

        ],
        [
            lambda funcs, start_x: gauss_newton(start_x, funcs, store_points=True),
            lambda funcs, start_x: bfgs(lambda x: sum([funcs[i](x) ** 2] for i in range(len(funcs))), start_x,
                                        store_points=True)
        ],
        ["Gauss Newton", "BFGS"]
    )


excel_saver = ExcellSaver()
# gauss_newton_vs_prev()
# dogleg_vs_prev()
# bfgs_vs_prev()
# lbfgs_vs_prev()
# gauss_newton_vs_dogleg()
# gauss_newton_vs_bfgs()
# dogleg_vs_bfgs()
# excel_saver.create_excel("metrics.xlsx")
