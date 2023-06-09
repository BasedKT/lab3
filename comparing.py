import numpy as np

from linreg import sgd_handler, LinearRegression, poly_array, Methods, visualise_linear, LearningRate
from main import dogleg


def comparing():
    linregs_examples = [
        LinearRegression(poly_array([1, 2]), np.ones(2), np.array((-1, 3, 5, -4, 8)), np.array((3, 2, 5, 2, -1)))
    ]
    step = 0.01
    lr = lambda *args: step

    for linreg in linregs_examples:
        dogleg_points = dogleg(linreg.loss, linreg.W, store_points=True)
        visualise_linear(linreg.loss, dogleg_points, "Dogleg", "w_1", "w_2")
        for method in Methods:
            linreg.W = np.ones(2)
            linreg.W_points = [np.copy(linreg.W)]
            sgd_handler(linreg, lr, method, store_points=True)
            sgd_points = linreg.W_points
            visualise_linear(linreg.loss, sgd_points, method.name, "w_1", "w_2")

comparing()