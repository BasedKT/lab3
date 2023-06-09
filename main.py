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
            np.linalg.pinv(J.body) @ func_wrap(start_point, funcs))
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
# NOTE: придумать что делать если гессиан не положительно определенный

def hessian(f):
    def calc(x):
        B = np.asarray([[0. for _ in range(len(x))] for _ in range(len(x))])
        for i in range(len(x)):
            for j in range(len(x)):
                B[i][j] = derivative(lambda x_tmp: derivative(f, x_tmp, j), x, i)
        return B

    return calc


def min_point_by_trust_region_func(f, x_k, get_p, recalc_m, delta=1., delta_max=10., eps=0.001, eta=0.2,
                                   max_steps=1000, store_points=False):
    div_eps = 0.0000000000000000000000000000000000001
    g = grad(f)
    B = hessian(f)
    points = [np.copy(x_k)]
    for i in range(max_steps):
        p_k = get_p(f, g, B, x_k, delta)
        m_k = recalc_m(f, g, B, x_k)
        ro_k = (f(x_k) - f(x_k + p_k)) / (m_k(np.zeros(len(x_k))) - m_k(p_k) + div_eps)
        if ro_k < 0.25:
            delta *= 0.25
        else:
            if ro_k > 0.75 and abs(np.linalg.norm(p_k) - delta) < eps:
                delta = min(2 * delta, delta_max)
        if ro_k > eta:
            if abs(f(x_k) - f(x_k + p_k)) < eps:
                break
            x_k += p_k

            if store_points:
                points.append(np.copy(x_k))

    return points


def recalc_m_for_dogleg(f, g, B, x):
    return lambda p: f(x) + g(x).T @ p + 0.5 * p.T @ B(x) @ p


def get_p_by_dogleg(f, g, B, x_k, delta):
    g_comp = g(x_k)
    B_comp = B(x_k)
    p_b = -np.linalg.inv(B_comp) @ g_comp

    if np.linalg.norm(p_b) <= delta:
        return p_b

    p_u = -((g_comp.T @ g_comp) / (g_comp.T @ B_comp @ g_comp)) * g_comp

    if np.linalg.norm(p_u) <= delta:
        p_delt = p_b - p_u
        return p_delt * (delta / np.linalg.norm(p_delt))
    else:
        return -(g_comp * (delta / np.linalg.norm(g_comp)))


def dogleg(f, x_k, delta=1., delta_max=10., eps=0.001, eta=0.2, max_steps=1000, store_points=False):
    return min_point_by_trust_region_func(f, x_k, get_p_by_dogleg, recalc_m_for_dogleg, delta, delta_max, eps, eta,
                                          max_steps, store_points)


# print(min_point_by_trust_region_func(
#     lambda ab: sum([(ab[0] * np.sin(i[0]) + ab[1] * np.cos(i[0]) - i[1]) ** 2 for i in
#                     [[-1.8769384497209494, 0.3208500555232947], [-1.5192895330096263, 0.7772238982741027],
#                      [-6.608540870385862, 0.28694079206001677], [6.914561957007351, -0.06537633613384441],
#                      [8.625076710401945, -0.5439463209772875], [5.1481282311186085, 0.2954095885701955],
#                      [9.917397351785983, 0.6381880291402098], [-3.1196118783564897, 0.22963471498883092],
#                      [7.770694338086578, -1.2223477289499438], [7.055529363737438, -1.0278035114572843],
#                      [-2.9556342432708416, 0.4986001160756802], [-9.23375090675026, -0.11282009684632777],
#                      [5.8102049796332, -0.06311829101291822], [-9.545091375894588, -0.2831938836416489],
#                      [-5.826223513214055, -0.21697795539658382], [-4.267697868348746, -0.4320163598688562],
#                      [-2.8351567506352993, -0.09527386407734567], [5.921967856445107, 0.3454482081185837],
#                      [4.976381342007938, 0.3995511186698528], [-2.8383706094032597, 0.032189666747653],
#                      [-0.1558717909590719, -0.3605651138568633], [8.311655199531476, -0.8862195212623891],
#                      [3.636532932128752, 0.1657490592521948], [-1.8587270756994219, 0.26306012702413495],
#                      [-7.003595530144828, 0.20856244222042208], [8.145120641402237, -1.1946442481763837],
#                      [-2.6039220943613266, 0.40330638566489185], [2.030233984766891, -0.4039502977372882],
#                      [-9.30838826455463, 0.5484281018529541], [-7.515652514258974, 0.9175554805036741],
#                      [-8.641684557556337, 0.2487934650864125], [5.787830587963349, 0.12871259477323344],
#                      [-3.336251538807698, -0.5607635286354037], [3.2989626372915133, 0.39796008847194314],
#                      [8.475275481335945, -0.1898871866282117], [8.510307667083573, -0.0882847385541009],
#                      [0.9403860630101857, -0.869380633325813], [-4.455111931533011, -0.26243599166441045],
#                      [3.6809065421937444, 0.8459068272506574], [-7.794230889015365, 0.5593025190698748],
#                      [8.416566787343513, -1.0135768400576146], [4.366223474432299, 0.7386057618087559],
#                      [-6.6243137214974235, 0.6434261606159205], [-4.957818865828165, -0.6363246862790314],
#                      [-2.3214050706217666, 0.1692868068782456], [-4.897174977451943, -0.7208018565918584],
#                      [2.2294776894465222, -0.8908121153134009], [-7.420431574442363, 1.1435382771435183],
#                      [9.317542826940517, 0.32359792377633667], [8.893593454955838, -0.5974341794945599],
#                      [9.798233640129919, 0.7446134535699878], [-1.7736250254030885, 1.2349185846703152],
#                      [8.114721599613357, -0.33309164366738186], [-4.203422879134838, -1.039643397490626],
#                      [-5.715077994512772, -0.6199246111973729], [-5.65282123529991, -0.3865670549416059],
#                      [-0.02618847153689252, -0.4812735346323904], [-3.0927736450612198, 0.4523686568831471],
#                      [1.9578262661860517, -0.23184286846988866], [9.042305545556832, -0.45899493868218166],
#                      [7.603352299789417, -0.5196427622191989], [6.462315734493561, -0.28379994231167505],
#                      [8.800318805259355, -0.8747963357090377], [7.982956456534332, -0.5661667650894068],
#                      [8.493633976545535, -0.6398908864225873], [-8.507453813809088, 0.22818545293343862],
#                      [-4.099226195893689, -0.6539443934810435], [-7.094286935689331, 0.38487506670975546],
#                      [2.1427046885173766, -0.5534282300406761], [5.333041010106152, 0.6292728704907733],
#                      [0.853971993876435, -0.38527038899230626], [-7.599813304016814, 0.38500486127178035],
#                      [8.074980951655238, -0.2787983527212796], [-8.401307320784529, 0.8869791593547965],
#                      [-9.581188783561354, -0.2676071029736081], [-6.765753630338482, 0.11839890625882782],
#                      [-5.774334189744044, -0.7909181968145023], [3.6709428631692873, 0.22794758235437784],
#                      [-0.22985889707427987, -0.3357482577501226], [-3.896827986583693, -0.38176431125364696],
#                      [-3.0952683519302866, -0.27630151900996536], [8.476756429943777, -0.9397625333302705],
#                      [9.2252207202283, 0.0683457109929656], [7.553374460978816, -0.6306034006252805],
#                      [0.06777147014958373, -0.02743905089885279], [-1.3759622857076277, 1.012823853632875],
#                      [9.696276569505745, 0.17676875366448352], [-3.8802860593980393, -0.11476431685199606],
#                      [-8.536713201505501, 0.6811143352038459], [-6.727458139672818, 0.30791046507181113],
#                      [-6.1363822008460644, -0.38070205955231873], [-3.045580589524599, 0.09395640991513232],
#                      [3.9205003475302984, 0.8971066369532453], [2.3872729089907434, -0.17871475261304282],
#                      [1.004438213973069, -1.1464860519635782], [-2.611790306414994, 0.24870209231862977],
#                      [6.567977983346353, -0.014017950038410776], [4.700813066523368, 0.6630580629820589],
#                      [6.17015546377386, -0.42730277869509875], [9.79178198342726, 0.08931314018149278]]]),
#     np.array((10., 10.)),
#     get_p_by_dogleg, recalc_m_for_dogleg
# ))
#
# print(hessian(lambda x: (x[0] - 5.) * (x[0] - 5.) + (x[1] + 6.) * (x[1] + 6.))(np.array((10., 10.))))

wolfe_cond_template = lambda c1, c2, x, func, gk: lambda a, b: not (
        (func.value(x - ((a + b) / 2) * gk) <= (func.value(x) + c1 * ((a + b) / 2) * np.dot(gk, -gk))) and (
        np.dot(func.grad(x - ((a + b) / 2) * gk), -gk) >= c2 * np.dot(gk, -gk)))

wolfe_cond = lambda: ""


def dichotomy(func, a_1, a_2, eps, delt, is_wolfe=False):
    cond = lambda a, b: abs(a - b) >= eps
    if is_wolfe:
        cond = wolfe_cond
    while cond(a_1, a_2):
        new_a_1 = (a_1 + a_2) / 2 - delt
        new_a_2 = (a_1 + a_2) / 2 + delt
        fv1 = func.value(new_a_1)
        fv2 = func.value(new_a_2)
        if fv2 > fv1:
            a_2 = new_a_2
        elif fv2 < fv1:
            a_1 = new_a_1
        else:
            a_1 = new_a_1
            a_2 = new_a_2
    return (a_1 + a_2) / 2


def right_border_calc(func):
    right_start = 0.0000001
    zero = func.value(0)
    while zero >= func.value(right_start):
        right_start *= 2

    return right_start


def calc_min_by_bfgs(x, f):
    I = np.eye(len(x))
    H = I
    grad_prev = grad(f)(x)
    p = -H * grad_prev
    x_prev = x
    x += dichotomy(f, 0, r) * p
    # в ск засунуть разность нового х и переданного
    # в ук засунуть разность градиента при новом х и при старом           также сохранить градиент и икс текущие
    for i in range(10000):
        y_k = grad(f)(x) - grad(f)(x_prev)
        s_k = x - x_prev
        ro_k = 1 / (y_k.T * s_k)
        H = (I - ro_k @ s_k @ y_k.T) @ H @ (I - ro_k @ y_k @ s_k.T) + ro_k @ s_k @ s_k.T
        p = -H @ grad(f)(x)
        # x -= alpa()
