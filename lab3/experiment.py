from prettytable import PrettyTable
from itertools import *
from math import sqrt, ceil
from scipy import stats


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


# проверка гипотезы об однородности мат модели (критерий Кохрена)
# y - массив результатов опытов [[], [], []]
# Gt - табличное значение
def cochran_test(y, y_avg, Gt=0.18):
    N = len(y)
    s2 = [0 for i in range(N)]
    m = len(y[0])
    for i in range(N):
        for j in range(m):
            s2[i] += pow(y[i][j] - y_avg[i], 2)
        s2[i] /= (m-1)
    s2_max = max(s2)
    s2_sum = sum(s2)
    Gp = s2_max / s2_sum
    return s2_sum, Gp < Gt


# проверка стат значимости коэффициентов
# coefficients - проверяемые коэффициенты
# s2_sum - сумма всех дисперсий
# N - количество опытов (строк)
# m - количество опытов для одной строки
def student_test(coefficients, s2_sum, N, m):
    # дисперсия воспроизводимости
    s_mean = s2_sum / N
    s_deviation = sqrt(s_mean / (N * m))
    result = [0 for i in range(len(coefficients))]
    tt = stats.t(df=N * m).ppf(0.95)
    for i in range(len(coefficients)):
        if abs(coefficients[i]) / s_deviation > tt:
            result[i] = 1
    return result, sum(result)


# проверка адекватности уравнения
# y_calculated - вычисленное по формуле значение
# y_avg - среднее по опытам для одной строки
# N - количество опытов (строк)
# m - количество опытов для одной строки
# L - количество значимых коэффициентов
def f_test(y_calculated, y_avg, N, m, L, s2_sum, Ft=3.2):
    # дисперсия воспроизводимости
    s_mean = s2_sum / N
    # дисперсия адекватности
    sum_avg = 0
    for i in range(len(y_calculated)):
        tmp = pow(y_avg[i] - y_calculated[i], 2)
        sum_avg += tmp
    sum_avg = m * sum_avg / (N - L)
    q1 = N
    q2 = N - L
    if s_mean >= sum_avg:
        q1 = N - L
        q2 = N
    Fp = sum_avg / s_mean

    return Fp < Ft


# ДФЭ


# коэффициенты со звездочкой
def dfe_calculate_coefficients(y, table, index_list):
    result = []
    N = len(table)
    for indexes in index_list:
        sum = 0
        for i in range(N):
            mul = y[i]
            for elem in indexes:
                mul *= table[i][elem]
            sum += mul
        result.append(sum / N)
    return result


# генерирующие соотношения
# x5 = x1x3
# x6 = x2x4
# x7 = x1x3x4
# x8 = x2x3x4
def dfe_get_x_row_all_linear(x_row: list):
    result = []
    # result.append(x_row[1] * x_row[2])
    assert len(x_row) == 5
    result.append(x_row[1] * x_row[3])
    result.append(x_row[2] * x_row[4])
    result.append(x_row[1] * x_row[3] * x_row[4])
    result.append(x_row[2] * x_row[3] * x_row[4])
    return result


# трасформация плана эксперимента из пфэ в дфэ с учетом генерирующих соотношений
# table - план эксперимента пфэ
# number_of_vars - исходное количество факторов
# k - показатель кратности
def dfe_transform_table(table, number_of_vars, k):
    dfe_plan = get_plan_matrix(number_of_vars - k)
    for row in dfe_plan:
        tmp = dfe_get_x_row_all_linear(row)
        row += tmp
    return dfe_plan


# генерация всех x в упрощенной модели
def dfe_get_all_x(table, index_list):
    result = []
    for row in table:
        new_row = row.copy()
        for indexes in index_list[9:]:
            mul = 1
            for elem in indexes:
                mul *= row[elem]
            new_row.append(mul)
        result.append(new_row)
    return result


# подсчет значения y в упрощенной модели (частично-нелинейной)
def dfe_calculate_nonlinear(coefficients, table, index_list):
    result = []
    for i in range(len(table)):
        sum = 0
        for j in range(len(coefficients)):
            mul = coefficients[j]
            for elem in index_list[j]:
                mul *= table[i][elem]
            sum += mul
        result.append(sum)
    return result


def get_coefficients_row_str(row):
    result = []
    for coefficient in row:
        tmp = 'b'
        for elem in coefficient:
            tmp += str(elem)
        result.append(tmp)
    return result


def get_coefficients_index_table(number_of_vars):
    x = [i for i in range(1, number_of_vars + 1)]
    combs = get_comb(x)
    all_coefficients = {'b0': 0}
    i = 1
    for comb in combs:
        coefficient = 'b'
        for elem in comb:
            coefficient += str(elem)
        all_coefficients[coefficient] = i
        i += 1
    return all_coefficients


# coefficients - коэффициенты со звездочкой
# index_list - список индексов в упрощенной модели
# mix_coefficients_table - таблица смешивания коэффициентов - указываются индексы коэффициентов чн модели
def dfe_split_coefficients(number_of_vars, coefficients, mix_coefficients_table):
    assert len(coefficients) == len(mix_coefficients_table)
    coefficients_index_table = get_coefficients_index_table(number_of_vars)
    result_coefficients = [0 for i in range(len(coefficients_index_table))]
    for i in range(number_of_vars + 1):
        result_coefficients[i] = coefficients[i]
    for i in range(number_of_vars + 1, len(coefficients)):
        str_row = get_coefficients_row_str(mix_coefficients_table[i])
        coefficients_indexes = []
        for elem in str_row:
            coefficients_indexes.append(coefficients_index_table[elem])
        for index in coefficients_indexes:
            result_coefficients[index] = coefficients[i] / len(mix_coefficients_table[i])

    return result_coefficients


# вспомогательная
def split_row(row, length):
    result = []
    for i in range(ceil(len(row) / length)):
        start = i * length
        end = start + length
        if end > len(row):
            end = len(row)
        result.append(row[start:end])
    return result


# тестовые данные
# коэффициенты в упрощенной модели
# index_list = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [1, 2], [1, 4], [1, 6],
#               [1, 7], [1, 8], [2, 3], [2, 5]]

# таблица смешивания коэффициентов
# mix_coefficients_table = [
#     [[0], [1, 3, 5], [2, 4, 6], [4, 5, 7], [3, 6, 8]],
#     [[1], [3, 5], [3, 4, 7], [2, 7, 8], [5, 6, 8]],
#     [[2], [4, 6], [3, 4, 8], [1, 7, 8], [5, 6, 7]],
#     [[3], [1, 5], [1, 4, 7], [2, 4, 8], [6, 8]],
#     [[4], [2, 6], [1, 3, 7], [2, 3, 8], [5, 7]],
#     [[5], [1, 3], [4, 7], [2, 6, 7], [1, 6, 8]],
#     [[6], [2, 4], [3, 8], [2, 5, 7], [1, 5, 8]],
#     [[7], [1, 3, 4], [4, 5], [1, 2, 8], [2, 5, 6]],
#     [[8], [2, 3, 4], [3, 6], [1, 2, 7], [1, 5, 6]],
#     [[1, 2], [2, 3, 5], [1, 4, 6], [4, 5, 8], [3, 6, 7]],
#     [[1, 4], [3, 4, 5], [1, 2, 6], [3, 7], [1, 5, 7], [2, 5, 8], [6, 7, 8]],
#     [[1, 6], [3, 5, 6], [1, 2, 4], [2, 3, 7], [1, 3, 8], [5, 8], [4, 7, 8]],
#     [[1, 7], [3, 5, 7], [3, 4], [1, 4, 5], [2, 3, 6], [2, 8], [4, 6, 8]],
#     [[1, 8], [3, 5, 8], [2, 4, 5], [1, 3, 6], [2, 7], [5, 6], [4, 6, 7]],
#     [[2, 3], [1, 2, 5], [3, 4, 6], [4, 8], [1, 6, 7], [2, 6, 8], [5, 7, 8]],
#     [[2, 5], [1, 2, 3], [4, 5, 6], [2, 4, 7], [1, 4, 8], [6, 7], [3, 7, 8]]
# ]


# index_list = [[0], [1], [2], [3], [4], [1, 3], [2, 3], [3, 4]]
# table = get_plan_matrix(4)
# y = [6.49, 4.44, -7.47, 2.53, 0.49, 2.50, 2.48, 16.52]
# new_table = dfe_transform_table(table, 4, 1)
# coefficients = dfe_calculate_coefficients(y, new_table, index_list)
# print(coefficients)
# print(dfe_calculate_nonlinear(coefficients, new_table, index_list))


# порядок действий ПФЭ:
# определяем количество факторов - number_of_vars;
# строим план эксперимента - plan = get_plan_matrix(number_of_vars)
#         - план эксперимента содержит только одиночные x, начиная с x0;
# чтобы получить значение всех x - get_all_x(plan);
# проводим эксперименты в количестве 2^number_of_vars - y:list;
# если для каждой строки плана проводится по одному эксперименту, то это значение и будет элементом списка y,
#         но если для каждой строки проводится несколько экспериментов, то в списке y будет среднее;
# считаем коэффициенты для уравнения регрессии (считаются сразу все - для частично-нелинейной)
#         - coeffs = calculate_coefficients(plan, y);
# считаем значение уравнения линейного (для каждой строки плана) - calculate(coeffs, plan);
# считаем значение уравнения частично-нелинейного(для каждой строки плана) - calculate_nonlinear(coeffs, plan);
# чтобы проверить гипотезу об однородности мат. модели - проверяем критерием Кохрена:
#         - s2_sum, cohchran_ok = cochran_test(y_full, y, 0.27),
#             где y_full - значения всех проведенных экспериментов ддля каждой строки плана;
# чтобы проверить гипотезу о стат. значимости полученных коэффициентов - проверяем критерием Стьюдента:
#         - coeff_ok, coeff_count = student_test(coeffs, s2_sum, len(y), test_count),
#             где test_count - количество экспериментов, проводимых для каждой строки плана,
#                 coeff_ok - массив 0 и 1, если 1, то коэффициент статистически значимый, иначе - 0;
# чтобы проверить гипотезу об адекватности мат. модели - проверяем критерием Фишера:
#         eq_ok = f_test(nonlinear, y, len(y), test_count, coeff_count, s2_sum, 2.4);


# порядок действий ДФЭ:
# определяем количество факторов - number_of_vars;
# определяем показатель кратности (численно равен кол-ву генерирующих соотношений) - k;
# определяем список индексов коэффициентов/факторов в упрощенной модели - index_list;
# строим план эксперимента - plan = get_plan_matrix(number_of_vars)
#         - план эксперимента содержит только одиночные x, начиная с x0;
# трансформируем матрицу исходя из генерирующих соотношений (их надо указать в функции dfe_get_x_row_all_linear),
#         при этом второстепенные факторы должны идти в конце (например, x1, x2, x3 - основные, x4 - второстепенный)
#             - new_plan = dfe_transform_table(plan, number_of_vars, k);
# чтобы получить значение всех x - dfe_get_all_x(plan, index_list);
# проводим эксперименты в количестве 2^(number_of_vars - k) - y:list (значения факторов в соответствии с новым планом);
# считаем коэффициенты для уравнения регрессии (считаются сразу все - для частично-нелинейной)
#         - coeffs = dfe_calculate_coefficients(new_plan, y, index_list);
# считаем значение уравнения линейного (для каждой строки плана) - calculate(coeffs, new_plan);
# считаем значение уравнения частично-нелинейного(для каждой строки плана)
#         - dfe_calculate_nonlinear(coeffs, new_plan, index_list);
# чтобы получить коэффициенты исходной математической модели
#         - full_coefficients = dfe_split_coefficients(number_of_vars, coefficients, mix_coefficients_table)
# проверить гипотезы можно также, как и в ПФЭ.
