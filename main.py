import math

import numpy as np


def derivative(f, x, i, delt=0.01):
    x_c = np.copy(x)
    x_c[i] += delt
    first_val = f(x_c)
    x_c[i] -= 2 * delt
    second_val = f(x_c)
    x_c[i] += delt
    return (first_val - second_val) / (2 * delt)


def grad(f, delt=0.01):
    def grad_calc(x):
        array = []
        for i in range(len(x)):
            array.append(derivative(f, x, i, delt))
        return np.array(array)

    return grad_calc


class Jacobian:
    # init_funcs - массив лямбд
    # arity - int, арность функций
    def __init__(self, funcs, arity):
        self.funcs = funcs
        self.eps = 0.01
        self.body = np.asarray([[0. for _ in range(arity)] for _ in range(len(funcs))])

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
    for i in range(10000):
        J.calc_body(start_point)
        if i % 100 == 0: print(start_point)
        # tmp_matrix = J.get_transposed_body() @ J.get_body()
        diff = np.asarray(start_point) - np.asarray(
            np.linalg.pinv(J.get_body()) @ func_wrap(start_point, funcs))
        start_point = diff


def func1(args):
    return args[0]


def func2(args):
    return args[0] ** 2


def func3(args):
    return math.cos(args[0]) + 1.0


# min_point_by_gauss_newton([1090990.9090], [func1, func2])
# print('\n')
# min_point_by_gauss_newton([919.0], [func3])

# print([[1., 2.], [0., 0.]] @ np.linalg.pinv([[1., 2.], [0., 0.]]))

# =======================================================DOG=LEG=========================================================
# придумать что делать если гессиан не положительно определенный

def hessian(f):
    def calc(x):
        B = np.asarray([[0. for _ in range(len(x))] for _ in range(len(x))])
        for i in range(len(x)):
            for j in range(len(x)):
                B[i][j] = derivative(lambda x_tmp: derivative(f, x_tmp, i), x, j)
        return B

    return calc


def min_point_by_trust_region_func(f, x_k, get_p, recalc_m, delta=1., delta_max=10., eps=0.001, eta=0.2,
                                   max_steps=1000):
    g = grad(f)
    B = hessian(f)(x_k)
    for i in range(max_steps):
        print(x_k)
        p_k = get_p(f, g, B, x_k, delta)
        m_k = recalc_m(f, g, B, x_k)
        ro_k = (f(x_k) - f(x_k + p_k)) / (m_k(np.zeros(len(x_k))) - m_k(p_k))
        if ro_k < 0.25:
            delta *= 0.25
        else:
            if ro_k > 0.75 and abs(np.linalg.norm(p_k) - delta) < eps:
                delta = min(2 * delta, delta_max)
        if ro_k > eta:
            if abs(f(x_k) - f(x_k + p_k)) < eps:
                break
            x_k += p_k

    return x_k


def recalc_m_for_dogleg(f, g, B, x):
    return lambda p: f(x) + g(x).T @ p


def get_p_by_dogleg(f, g, B, x_k, delta):
    g_comp = g(x_k)
    B_comp = B
    p_b = -np.linalg.inv(B_comp) @ g_comp

    if np.linalg.norm(p_b) <= delta:
        return p_b

    p_u = -((g_comp.T @ g_comp) / (g_comp.T @ B_comp @ g_comp)) * g_comp

    # l = 1.
    # r = 2.
    # p_func = lambda teta: p_u + (teta - 1.) * (p_b - p_u)
    # eps = 0.001
    # norm = np.linalg.norm(p_func(l))
    # while delta - norm > eps or norm > delta:
    #     mid = (l + r) / 2
    #     mid_norm = np.linalg.norm(p_func(mid))
    #     if mid_norm - delta > eps:
    #         r = mid
    #     else:
    #         l = mid
    #         norm = mid_norm

    a = np.linalg.norm(p_b - p_u) ** 2
    b = 2 * np.dot(p_u, p_b - p_u)
    c = np.linalg.norm(p_u) ** 2 - delta * delta
    # a = 0.
    # b = 0.
    # c = 0.
    # for i in range(len(x_k)):
    #     a += (p_b[i] - p_u[i]) ** 2
    #     b += (p_b[i] - p_u[i]) * p_u[i]
    #     c += (p_u[i]) ** 2
    # b *= 2
    # c -= delta ** 2
    D = b ** 2 - 4. * a * c
    if abs(D) <= 0.009: D = 0
    print(a)
    print(b)
    print(c)
    print(D)
    x1 = (-b + math.sqrt(D)) / 2.
    x2 = (-b - math.sqrt(D)) / 2.
    tau1 = x1 + 1
    tau2 = x2 + 1
    real_tau = tau1 if (0 <= tau1 <= 1) or (1 <= tau1 <= 2) else tau2
    return real_tau * p_u if (0 <= real_tau <= 1) else p_u + (real_tau - 1) * (p_b - p_u)


print(min_point_by_trust_region_func(
    lambda x: (x[0] - 5.) * (x[0] - 5.) + (x[1] + 6.) * (x[1] + 6.),
    np.array((10., 10.)),
    get_p_by_dogleg, recalc_m_for_dogleg
))

# print(hessian(lambda x: (x[0] - 5.) * (x[0] - 5.) + (x[1] + 6.) * (x[1] + 6.))(np.array((10., 10.))))
