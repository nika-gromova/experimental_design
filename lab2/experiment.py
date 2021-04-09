from prettytable import PrettyTable
from itertools import *


# план эксперимента
# number_of_vars - количество факторов
def get_plan_matrix(number_of_vars):
    number_of_experiments = pow(2, number_of_vars)
    table = [[0 for i in range(number_of_vars + 1)] for i in range(number_of_experiments)]
    for i in range(number_of_experiments):
        table[i][0] = 1
    for j in range(1, number_of_vars + 1):
        step = pow(2, j - 1)
        value = 1
        table[0][j] = 1
        for i in range(1, number_of_experiments):
            if i % step == 0:
                value = -value
            table[i][j] = value
    return table


# все комбинации без повторений элементов из массива x
def get_comb(x):
    result = []
    for i in range(1, len(x) + 1):
        for x_comb in combinations(x, i):
            result.append(x_comb)
    return result


# вернуть массив всех коэффициентов x по известным переменным (для одной строки матрицы)
def get_x_row_nonlinear(x):
    result = []
    result.append(x[0])
    for i in range(1, len(x)):
        for x_comb in combinations(x[1:], i):
            tmp = 1
            for item in x_comb:
                tmp *= item
            result.append(tmp)
    return result


# все коэффициенты x
def get_all_x(table):
    result = []
    for i in range(len(table)):
        result.append(get_x_row_nonlinear(table[i]))
    return result


# линейная регрессия
def calculate(coefficients, table):
    coefficient_count = len(table[0])
    result = [0 for i in range(len(table))]
    for i in range(len(table)):
        for j in range(coefficient_count):
            result[i] += table[i][j] * coefficients[j]
    return result


# частично нелинейная
def calculate_nonlinear(coefficients, table):
    coefficient_count = len(coefficients)
    result = [0 for i in range(len(table))]
    for i in range(len(table)):
        x_row = get_x_row_nonlinear(table[i])
        assert len(x_row) == coefficient_count
        for j in range(coefficient_count):
            result[i] += x_row[j] * coefficients[j]
    return result


# коэффициенты уравнения регрессии
def calculate_coefficients(table, y):
    coefficients = []

    first_coefficient = 0
    for i in range(len(table)):
        first_coefficient += table[i][0] * y[i]
    first_coefficient /= len(table)
    coefficients.append(first_coefficient)

    x_count = len(table[0]) - 1
    x = [i for i in range(1, x_count + 1)]

    for i in range(1, x_count + 1):
        for x_comb in combinations(x, i):
            result = 0
            for row in range(len(table)):
                tmp_result = 1
                for item in x_comb:
                    tmp_result *= table[row][item]
                result += tmp_result * y[row]
            result /= len(table)
            coefficients.append(result)

    return coefficients


# преобразование факторов
def scale_x(variance_param, value):
    param_min = variance_param[0]
    param_max = variance_param[1]
    return 2 * (value - param_min) / (param_max - param_min) - 1


# coefficients - вычисленные b0,b1,...;
# params - сами кастомные факторы
# variance_params - ограничения [min, max]
# вычисление в пргоизвольной точке факторного пространства
def calculate_custom(coefficients, params, variance_params):
    scaled_params = []
    scaled_params.append(1)
    for i in range(len(params)):
        scaled_params.append(scale_x(variance_params[i], params[i]))
    result_linear = calculate(coefficients, [scaled_params])
    result_nonlinear = calculate_nonlinear(coefficients, [scaled_params])
    return scaled_params, result_linear, result_nonlinear
