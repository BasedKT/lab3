import math

import numpy as np


class Jacobian:
    # init_funcs - массив лямбд
    # arity - int, арность функций
    def __init__(self, init_funcs, arity):
        self.funcs = init_funcs
        self.eps = 0.01
        self.body = np.asarray([[0. for _ in range(arity)] for _ in range(len(init_funcs))])

    # point - точка в виде массива
    def calc_body(self, point):
        for i in range(len(self.funcs)):
            for j in range(len(point)):
                prev = point[j]
                point[j] += self.eps
                minuend_func = self.funcs[i]
                minuend = minuend_func(point)
                point[j] = prev
                subtrahend = self.funcs[i](point)
                self.body[i][j] = (minuend - subtrahend) / self.eps

    def get_body(self):
        return self.body

    def get_transposed_body(self):
        return np.transpose(self.body)


# point - точка в виде массива
# funcs - массив лямбд
def func_wrap(point, funcs):
    result = np.array([0. for _ in range(len(funcs))])
    for i in range(len(funcs)):
        result[i] = funcs[i](point)
    return result


# start_point - точка в виде массива
# funcs - массив лямбд
def min_point_by_gauss_newton(start_point, funcs):
    start_point = np.asarray(start_point)
    J = Jacobian(funcs, len(start_point))
    for i in range(1000):
        J.calc_body(start_point)
        if i % 100 == 0: print((start_point[0]))
        tmp_matrix = J.get_transposed_body() @ J.get_body()
        diff = np.asarray(start_point) - np.asarray(
            np.linalg.inv(tmp_matrix) @ J.get_transposed_body() @ func_wrap(start_point, funcs))
        start_point = diff


def func1(args):
    return args[0]


def func2(args):
    return args[0] ** 2


def func3(args):
    return math.cos(args[0]) + 1.0


min_point_by_gauss_newton([1090990.9090], [func1, func2])
print('\n')
min_point_by_gauss_newton([919.0], [func3])
