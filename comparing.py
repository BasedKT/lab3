import random
from time import time

import numpy as np

import tracemalloc

from linreg import sgd_handler, Methods, visualise_linear, visualise_approximation, gen_linear_reg
from main import dogleg, gauss_newton, bfgs
from excel import ExcellSaver


class Metrics:
    def __init__(self, mem, steps, time):
        self.mem = mem
        self.points = steps
        self.time = time


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


def comparing(linreg_func, title, visualize_prev_methods=True):
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
        visualise_linear(linreg, title, "w_1", "w_2")
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
                visualise_linear(linreg, method.name, "w_1", "w_2")
                visualise_approximation(linreg, method)


def gauss_newton_comparing():
    global excel_saver

    def calc_points(start_x, linreg):
        funcs = np.array([lambda W, i=i: np.dot(linreg.T[i], W) - linreg.Y[i] for i in range(len(linreg.X))])
        return gauss_newton(start_x, funcs, store_points=True)

    excel_saver.add_new_sheet(["method", "count steps", "loss", "memory", "time"], "Gauss-Newton")
    comparing(calc_points, "Gauss-Newton", visualize_prev_methods=True)


def dogleg_comparing():
    global excel_saver

    def calc_points(start_x, linreg):
        return dogleg(linreg.loss, start_x, store_points=True)

    excel_saver.add_new_sheet(["method", "count steps", "loss", "memory", "time"], "Dogleg")
    comparing(calc_points, "Dogleg", visualize_prev_methods=True)


def bfgs_comparing():
    global excel_saver

    def calc_points(start_x, linreg):
        return bfgs(linreg.loss, start_x, store_points=True)

    excel_saver.add_new_sheet(["method", "count steps", "loss", "memory", "time"], "BFGS")
    comparing(calc_points, "BFGS", visualize_prev_methods=True)


excel_saver = ExcellSaver()
gauss_newton_comparing()
dogleg_comparing()
bfgs_comparing()
excel_saver.create_excel("metrics.xlsx")
