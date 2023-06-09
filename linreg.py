import random
from math import exp

import matplotlib.pyplot as plt

import numpy as np
from enum import Enum

Methods = Enum('Methods', ['Classic', 'Momentum', 'AdaGrad', 'RMSprop', 'Adam', 'Nesterov'])
Regularization = Enum('Regularization', ['WithoutRegularization', 'L1', 'L2', 'Elastic'])
LearningRate = Enum('LearningRate', ['Const'])
LearningRateScheduling = Enum('LearningRateScheduling', ['Classic', 'Stepwise', 'Exponential'])


def sign(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1


class LinearRegression:
    def __init__(self, T, W, X, Y, regularization=Regularization.WithoutRegularization, l1=0.1, l2=0.1):
        self.T = np.array([T[i % len(T)](X[i // len(T)]) for i in range(len(T) * len(X))]).reshape(len(X), len(T))
        self.W = W
        self.X = X
        self.Y = Y
        self.regularization = regularization
        self.l1 = l1
        self.l2 = l2
        self.W_points = [np.copy(self.W)]
        self.loss_values = [self.loss(self.W)]

    def loss(self, W_Arg, is_avarage=False):
        val = sum([(np.dot(self.T[i], W_Arg) - self.Y[i]) ** 2 for i in range(len(self.X))])
        match self.regularization:
            case Regularization.L1:
                val += self.l1 * sum([abs(w) for w in self.W])
            case Regularization.L2:
                val += self.l2 * sum([w ** 2 for w in self.W])
            case Regularization.Elastic:
                val += (self.l1 * sum([abs(w) for w in self.W])) + (self.l2 * sum([w ** 2 for w in self.W]))

        return val / len(self.X) if is_avarage else val

    def grad_by_components(self, index_components, W_Arg):
        grad_with_batch = np.zeros(len(W_Arg))
        for i in index_components:
            grad_with_batch += (2 * (np.dot(self.T[i], W_Arg) - self.Y[i]) * self.T[i])
        match self.regularization:
            case Regularization.L1:
                grad_with_batch += self.l1 * np.array([sign(w) for w in self.W])
            case Regularization.L2:
                grad_with_batch += self.l2 * 2 * self.W
            case Regularization.Elastic:
                grad_with_batch += (self.l1 * np.array([sign(w) for w in self.W])) + (self.l2 * 2 * self.W)

        return grad_with_batch

    def analytical_solution(self):
        return (np.linalg.inv(np.transpose(self.T) @ self.T) @ np.transpose(self.T)) @ self.Y


def sgd(lin_reg, lr, lrs, batch, max_num_of_step, beta_1, beta_2, eps_adam, is_corr_beta_1=True,
        is_corr_beta_2=True, is_nesterov=False, is_adagrad=False, store_points=False):
    i = -1
    V = np.zeros(len(lin_reg.W))
    S = np.zeros(len(lin_reg.W))
    lrs_func = lrs_handler(lrs)
    while True:
        i += 1

        components = [(i * batch + j) % len(lin_reg.X) for j in range(batch)]
        cur_w = lin_reg.W
        grad_with_batch = lin_reg.grad_by_components(components, cur_w)

        alpha = lrs_func(lr(lambda a: lin_reg.loss(lin_reg.W - a * grad_with_batch)), (i * batch) // len(lin_reg.X))
        if is_nesterov:
            cur_w -= alpha * beta_1 * V
            grad_with_batch = lin_reg.grad_by_components(components, cur_w)

        V = (beta_1 * V) + (1 - beta_1) * grad_with_batch
        S = (beta_2 * S) + (1 - beta_2) * (grad_with_batch ** 2) if ~is_adagrad else (S + grad_with_batch ** 2)
        V_norm = V / (1 - beta_1 ** (i + 1)) if is_corr_beta_1 else V
        S_norm = S / (1 - beta_2 ** (i + 1)) if is_corr_beta_2 else S

        lin_reg.W = lin_reg.W - alpha * (V_norm / ((S_norm + eps_adam) ** 0.5))

        loss_W = lin_reg.loss(lin_reg.W)
        if store_points:
            lin_reg.W_points.append(np.copy(lin_reg.W))
        lin_reg.loss_values.append(loss_W)
        if i >= max_num_of_step:
            break

    return i


def sgd_handler(lin_reg, lr, method, lrs=LearningRateScheduling.Classic, batch=1, beta_1=0.9, beta_2=0.999,
                eps_adam=10 ** -8, max_num_of_step=5000, store_points=False):
    match method:
        case Methods.Classic:
            return sgd(lin_reg, lr, lrs, batch, max_num_of_step, beta_1=0., beta_2=1., eps_adam=1,
                       is_corr_beta_1=False, is_corr_beta_2=False, store_points=store_points)
        case Methods.Momentum:
            return sgd(lin_reg, lr, lrs, batch, max_num_of_step, beta_1, beta_2=1., eps_adam=1,
                       is_corr_beta_1=False, is_corr_beta_2=False, store_points=store_points)
        case Methods.AdaGrad:
            return sgd(lin_reg, lr, lrs, batch, max_num_of_step, beta_1=0., beta_2=0.5, eps_adam=eps_adam,
                       is_corr_beta_1=False, is_corr_beta_2=False, is_adagrad=True, store_points=store_points)
        case Methods.RMSprop:
            return sgd(lin_reg, lr, lrs, batch, max_num_of_step, beta_1=0., beta_2=beta_2, eps_adam=eps_adam,
                       is_corr_beta_1=False, store_points=store_points)
        case Methods.Adam:
            return sgd(lin_reg, lr, lrs, batch, max_num_of_step, beta_1, beta_2, eps_adam,
                       store_points=store_points)
        case Methods.Nesterov:
            return sgd(lin_reg, lr, lrs, batch, max_num_of_step, beta_1, beta_2=1., eps_adam=1,
                       is_corr_beta_1=False, is_corr_beta_2=False, is_nesterov=True, store_points=store_points)


def lrs_exp(decay):
    return lambda lr, epoch: lr * exp(-decay * epoch)


def lrs_step(decay, epoch_update):
    return lambda lr, epoch: lr * (decay ** (epoch // epoch_update))


def lrs_handler(lrs, epoch_update=20):
    match lrs:
        case LearningRateScheduling.Classic:
            return lambda lr, epoch: lr
        case LearningRateScheduling.Stepwise:
            return lrs_step(0.75, epoch_update)
        case LearningRateScheduling.Exponential:
            return lrs_exp(0.1)


def visualise_linear(f, points, title, x_label="x", y_label="y"):
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


def poly_array(coeffs):
    return np.array([lambda x, i=i: coeffs[i] * (x ** i) for i in range(len(coeffs))])


def poly(poly_arr):
    return lambda x: sum([poly_arr[i](x) for i in range(len(poly_arr))])


def generate_data(num_of_points, dimension, coeffs_left, coeffs_right, x_left, x_right, deviation, own_coeffs=None):
    coeffs = own_coeffs
    if coeffs == None:
        coeffs = np.array([float(random.randint(coeffs_left, coeffs_right)) for i in range(dimension + 1)])

    X = [random.uniform(x_left, x_right) for _ in range(num_of_points)]
    Y = [poly(poly_array(coeffs))(X[i]) + random.uniform(-deviation, +deviation) for i in range(num_of_points)]

    return [np.array(X), np.array(Y), coeffs]


def gen_linear_reg(dimension, num_of_points, coeffs_left, coeffs_right, x_left, x_right, deviation, own_coeffs=None):
    T = np.array(poly_array(np.ones(dimension + 1)))
    X, Y, coeffs = generate_data(num_of_points, dimension, coeffs_left, coeffs_right, x_left, x_right, deviation,
                                 own_coeffs)
    W = np.ones(len(coeffs))

    return LinearRegression(T, W, X, Y)
