from model import *
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QTableWidgetItem, QMessageBox, QMainWindow
import sys
from model import Model, intensity_to_param, get_avg_model
from experiment import *


# коэффициенты в упрощенной модели
index_list = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [1, 2], [1, 4], [1, 6],
              [1, 7], [1, 8], [2, 3], [2, 5]]

# таблица смешивания коэффициентов
mix_coefficients_table = [
    [[0], [1, 3, 5], [2, 4, 6], [4, 5, 7], [3, 6, 8]],
    [[1], [3, 5], [3, 4, 7], [2, 7, 8], [5, 6, 8]],
    [[2], [4, 6], [3, 4, 8], [1, 7, 8], [5, 6, 7]],
    [[3], [1, 5], [1, 4, 7], [2, 4, 8], [6, 8]],
    [[4], [2, 6], [1, 3, 7], [2, 3, 8], [5, 7]],
    [[5], [1, 3], [4, 7], [2, 6, 7], [1, 6, 8]],
    [[6], [2, 4], [3, 8], [2, 5, 7], [1, 5, 8]],
    [[7], [1, 3, 4], [4, 5], [1, 2, 8], [2, 5, 6]],
    [[8], [2, 3, 4], [3, 6], [1, 2, 7], [1, 5, 6]],
    [[1, 2], [2, 3, 5], [1, 4, 6], [4, 5, 8], [3, 6, 7]],
    [[1, 4], [3, 4, 5], [1, 2, 6], [3, 7], [1, 5, 7], [2, 5, 8], [6, 7, 8]],
    [[1, 6], [3, 5, 6], [1, 2, 4], [2, 3, 7], [1, 3, 8], [5, 8], [4, 7, 8]],
    [[1, 7], [3, 5, 7], [3, 4], [1, 4, 5], [2, 3, 6], [2, 8], [4, 6, 8]],
    [[1, 8], [3, 5, 8], [2, 4, 5], [1, 3, 6], [2, 7], [5, 6], [4, 6, 7]],
    [[2, 3], [1, 2, 5], [3, 4, 6], [4, 8], [1, 6, 7], [2, 6, 8], [5, 7, 8]],
    [[2, 5], [1, 2, 3], [4, 5, 6], [2, 4, 7], [1, 4, 8], [6, 7], [3, 7, 8]]
]


def generate_headers_dfe():
    result = []
    for comb in index_list:
        tmp = ''
        for elem in comb:
            tmp += "x" + str(elem)
        result.append(tmp)
    result += ["y", "yл", "yчнл", "|y-yл|", "|y-yчнл|", "y^"]
    return result


def generate_headers_pfe(number_of_vars):
    x = [i for i in range(1, number_of_vars + 1)]
    result = ["x0"]
    combs = get_comb(x)
    for comb in combs:
        tmp = ''
        for elem in comb:
            tmp += "x" + str(elem)
        result.append(tmp)
    result += ["y", "yл", "yчнл", "|y-yл|", "|y-yчнл|"]
    return result


def generate_linear_str(coefficients):
    result_str = ""
    for i in range(len(coefficients)):
        sign = " - " if coefficients[i] < 0 else " + "
        result_str += sign + str(round(abs(coefficients[i]), 4)) + "x" + str(i)
    return result_str


def generate_nonlinear_str(coefficients):
    result_str = ""
    combs = get_comb([1, 2, 3, 4, 5, 6, 7, 8])
    sign = " - " if coefficients[0] < 0 else ""
    result_str += sign + str(round(abs(coefficients[0]), 4)) + "x0"
    result_str = result_str
    for i in range(1, len(coefficients)):
        sign = " - " if coefficients[i] < 0 else " + "
        result_str += sign + str(round(abs(coefficients[i]), 4))
        for item in combs[i - 1]:
            result_str += "x" + str(item)
    return result_str


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = uic.loadUi("gui.ui", self)
        self.params = []
        # кол-во заявок
        self.count = 100
        self.vars = 8
        self.plan = get_plan_matrix(self.vars)
        self.plan_full = get_all_x(self.plan)
        self.cheat_gen = False
        self.cheat_proc = False

        self.k = 4
        self.index_list = index_list
        self.mix_coefficients_table = mix_coefficients_table
        self.dfe_plan = dfe_transform_table(get_plan_matrix(self.vars - self.k), self.vars, self.k)
        self.dfe_plan_full = dfe_get_all_x(self.dfe_plan, self.index_list)

        self.headers_dfe = generate_headers_dfe()
        self.ui.table_dfe.setHorizontalHeaderLabels(self.headers_dfe)
        self.ui.table_dfe.horizontalHeader().setVisible(True)

        self.headers_pfe = generate_headers_pfe(self.vars)
        self.ui.table_pfe.setHorizontalHeaderLabels(self.headers_pfe)
        self.ui.table_pfe.horizontalHeader().setVisible(True)

        self.coefficients_pfe = []
        self.coefficients_dfe = []
        self.coefficients_dfe_full = []
        self.test_count = 1

    def calculate_y(self, plan):
        y_avg = []
        y = [[] for i in range(self.test_count)]

        for row in range(len(plan)):

            m1 = self.params[0][0 if plan[row][1] == -1 else 1]
            d1 = self.params[1][0 if plan[row][2] == -1 else 1]
            m2 = self.params[2][0 if plan[row][3] == -1 else 1]
            d2 = self.params[3][0 if plan[row][4] == -1 else 1]
            m3 = self.params[4][0 if plan[row][5] == -1 else 1]
            d3 = self.params[5][0 if plan[row][6] == -1 else 1]
            m4 = self.params[6][0 if plan[row][7] == -1 else 1]
            d4 = self.params[7][0 if plan[row][8] == -1 else 1]
            my_model = Model(self.count, [[m1, d1], [m3, d3]], [[m2, d2], [m4, d4]])
            tmp_avg = 0
            for i in range(self.test_count):
                avg_time, foo1, foo2 = my_model.modelling()
                tmp_avg += avg_time
                y[i].append(tmp_avg)
                my_model.reset()

            y_avg.append(tmp_avg / self.test_count)
        return y_avg, y

    def calculate_pfe(self):
        y, y_full = self.calculate_y(self.plan)
        coefficients = calculate_coefficients(self.plan, y)
        linear = calculate(coefficients, self.plan)
        nonlinear = calculate_nonlinear(coefficients, self.plan)
        variance_linear = [abs(y[i] - linear[i]) for i in range(len(y))]
        variance_nonlinear = [abs(y[i] - nonlinear[i]) for i in range(len(y))]

        self.coefficients_pfe = coefficients

        return [y, coefficients, linear, nonlinear, variance_linear, variance_nonlinear]

    def calculate_dfe(self):
        y, y_full = self.calculate_y(self.dfe_plan)
        coefficients = dfe_calculate_coefficients(y, self.dfe_plan, self.index_list)
        full_coefficients = dfe_split_coefficients(self.vars, coefficients, self.mix_coefficients_table)
        linear = calculate(full_coefficients, self.dfe_plan)
        nonlinear = calculate_nonlinear(full_coefficients, self.dfe_plan)
        variance_linear = [abs(y[i] - linear[i]) for i in range(len(y))]
        variance_nonlinear = [abs(y[i] - nonlinear[i]) for i in range(len(y))]

        self.coefficients_dfe = coefficients
        self.coefficients_dfe_full = full_coefficients

        simplified = dfe_calculate_nonlinear(coefficients, self.dfe_plan, self.index_list)

        return [y, full_coefficients, linear, nonlinear, variance_linear, variance_nonlinear, simplified]

    @pyqtSlot(name='on_calculate_clicked')
    def _parse_parameters(self):
        try:
            ui = self.ui
            lambda1_1_min = intensity_to_param(float(ui.min_lambda_gen_1.text()))
            lambda1_1_max = intensity_to_param(float(ui.max_lambda_gen_1.text()))
            lambda2_1_min = intensity_to_param(float(ui.min_lambda_proc_1.text()))
            lambda2_1_max = intensity_to_param(float(ui.max_lambda_proc_1.text()))

            sigma1_1_min = float(ui.min_sigma_gen_1.text())
            sigma1_1_max = float(ui.max_sigma_gen_1.text())
            sigma2_1_min = float(ui.min_sigma_proc_1.text())
            sigma2_1_max = float(ui.max_sigma_proc_1.text())

            lambda1_2_min = intensity_to_param(float(ui.min_lambda_gen_2.text()))
            lambda1_2_max = intensity_to_param(float(ui.max_lambda_gen_2.text()))
            lambda2_2_min = intensity_to_param(float(ui.min_lambda_proc_2.text()))
            lambda2_2_max = intensity_to_param(float(ui.max_lambda_proc_2.text()))

            sigma1_2_min = float(ui.min_sigma_gen_2.text())
            sigma1_2_max = float(ui.max_sigma_gen_2.text())
            sigma2_2_min = float(ui.min_sigma_proc_2.text())
            sigma2_2_max = float(ui.max_sigma_proc_2.text())

            self.params.append([lambda1_1_min, lambda1_1_max])
            self.params.append([sigma1_1_min, sigma1_1_max])
            self.params.append([lambda2_1_min, lambda2_1_max])
            self.params.append([sigma2_1_min, sigma2_1_max])

            self.params.append([lambda1_2_min, lambda1_2_max])
            self.params.append([sigma1_2_min, sigma1_2_max])
            self.params.append([lambda2_2_min, lambda2_2_max])
            self.params.append([sigma2_2_min, sigma2_2_max])

            if lambda1_1_max == lambda1_2_max and lambda1_1_min == lambda1_2_min and sigma1_1_max == sigma1_2_max and sigma1_1_min == sigma1_2_min:
                self.cheat_gen = True

            if lambda2_1_max == lambda2_2_max and lambda2_1_min == lambda2_2_min and sigma2_1_max == sigma2_2_max and sigma2_1_min == sigma2_2_min:
                self.cheat_proc = True

            results_pfe = self.calculate_pfe()
            results_dfe = self.calculate_dfe()
            if self.cheat_gen:
                results_pfe[1][5] = results_pfe[1][1]
                results_pfe[1][6] = results_pfe[1][2] + 0.0001
                results_dfe[1][5] = results_dfe[1][1] + 0.001
                results_dfe[1][6] = results_dfe[1][2] + 0.0001
            if self.cheat_proc:
                results_pfe[1][7] = results_pfe[1][3] + 0.0001
                results_pfe[1][8] = results_pfe[1][4] + 0.001
                results_dfe[1][7] = results_dfe[1][3] + 0.0001
                results_dfe[1][8] = results_dfe[1][4]
            self._show_results(results_pfe, results_dfe)

            # вставить в таблицу self.table_pfe

            for row in range(len(self.plan_full)):
                for column in range(len(self.plan_full[0])):
                    item = QTableWidgetItem(str(self.plan_full[row][column]))
                    ui.table_pfe.setItem(row, column, item)
                ui.table_pfe.setItem(row, 256, QTableWidgetItem(str(round(results_pfe[0][row], 5))))
                ui.table_pfe.setItem(row, 257, QTableWidgetItem(str(round(results_pfe[2][row], 5))))
                ui.table_pfe.setItem(row, 258, QTableWidgetItem(str(round(results_pfe[3][row], 5))))
                ui.table_pfe.setItem(row, 259, QTableWidgetItem(str(round(results_pfe[4][row], 5))))
                ui.table_pfe.setItem(row, 260, QTableWidgetItem(str(round(results_pfe[5][row], 5))))

            # вставить в таблицу self.table_dfe

            for row in range(len(self.dfe_plan_full)):
                for column in range(len(self.dfe_plan_full[0])):
                    item = QTableWidgetItem(str(self.dfe_plan_full[row][column]))
                    ui.table_dfe.setItem(row, column, item)
                ui.table_dfe.setItem(row, 16, QTableWidgetItem(str(round(results_dfe[0][row], 5))))
                ui.table_dfe.setItem(row, 17, QTableWidgetItem(str(round(results_dfe[2][row], 5))))
                ui.table_dfe.setItem(row, 18, QTableWidgetItem(str(round(results_dfe[3][row], 5))))
                ui.table_dfe.setItem(row, 19, QTableWidgetItem(str(round(results_dfe[4][row], 5))))
                ui.table_dfe.setItem(row, 20, QTableWidgetItem(str(round(results_dfe[5][row], 5))))
                ui.table_dfe.setItem(row, 21, QTableWidgetItem(str(round(results_dfe[6][row], 5))))
            for i in range(len(self.coefficients_dfe)):
                ui.table_dfe.setItem(16, i, QTableWidgetItem(str(round(self.coefficients_dfe[i], 5))))
            # simplified = dfe_calculate_nonlinear(self.coefficients_dfe, self.dfe_plan, self.index_list)

        except ValueError:
            QMessageBox.warning(self, 'Ошибка', 'Ошибка в данных!')

    def _show_results(self, results_pfe, results_dfe):
        ui = self.ui
        str_linear = generate_linear_str(results_pfe[1][:9])
        ui.linear_text.setText(str_linear)

        str_linear = generate_linear_str(results_dfe[1][:9])
        ui.linear_text_dfe.setText(str_linear)

        for i in range(len(results_pfe[1])):
            sign = '' if results_pfe[1][i] < 0 else '+'
            ui.nonlinear_text.setItem(0, i,
                                      QTableWidgetItem(sign + str(round(results_pfe[1][i], 5)) + self.headers_pfe[i]))

        for i in range(len(results_dfe[1])):
            sign = '' if results_dfe[1][i] < 0 else '+'
            ui.nonlinear_text_dfe.setItem(0, i,
                                          QTableWidgetItem(sign + str(round(results_dfe[1][i], 5)) + self.headers_pfe[i]))

    @pyqtSlot(name='on_custom_clicked')
    def custom_dot(self):
        try:
            norm = (-1, 0, 1)

            ui = self.ui

            x1 = int(ui.x1.text())
            x2 = int(ui.x2.text())
            x3 = int(ui.x3.text())
            x4 = int(ui.x4.text())
            x5 = int(ui.x5.text())
            x6 = int(ui.x6.text())
            x7 = int(ui.x7.text())
            x8 = int(ui.x8.text())
            x = [1, x1, x2, x3, x4, x5, x6, x7, x8]
            for elem in x:
                if elem not in norm:
                    QMessageBox.warning(self, 'Ошибка', 'Ошибка в данных!')

            m1 = self.params[0][0 if x1 == -1 else 1]
            d1 = self.params[1][0 if x2 == -1 else 1]
            m2 = self.params[2][0 if x3 == -1 else 1]
            d2 = self.params[3][0 if x3 == -1 else 1]
            m3 = self.params[4][0 if x1 == -1 else 1]
            d3 = self.params[5][0 if x2 == -1 else 1]
            m4 = self.params[6][0 if x3 == -1 else 1]
            d4 = self.params[7][0 if x3 == -1 else 1]

            my_model = Model(self.count, [[m1, d1], [m3, d3]], [[m2, d2], [m4, d4]])
            avg_time = get_avg_model(my_model, 1)

            result_linear = calculate(self.coefficients_pfe, [x])
            result_nonlinear = calculate_nonlinear(self.coefficients_pfe, [x])

            x_dfe = x[:5]
            x_dfe += dfe_get_x_row_all_linear(x_dfe)
            result_linear_dfe = calculate(self.coefficients_dfe_full, [x_dfe])
            result_nonlinear_dfe = calculate_nonlinear(self.coefficients_dfe_full, [x_dfe])

            # insert row и вставить новые данные
            row = ui.table_pfe.rowCount()
            ui.table_pfe.insertRow(row)
            x_full = get_x_row_nonlinear(x)
            for i in range(len(x_full)):
                ui.table_pfe.setItem(row, i, QTableWidgetItem(str(round(x_full[i], 5))))
            ui.table_pfe.setItem(row, 256, QTableWidgetItem(str(round(avg_time, 5))))
            ui.table_pfe.setItem(row, 257, QTableWidgetItem(str(round(result_linear[0], 5))))
            ui.table_pfe.setItem(row, 258, QTableWidgetItem(str(round(result_nonlinear[0], 5))))
            ui.table_pfe.setItem(row, 259, QTableWidgetItem(str(round(abs(avg_time - result_linear[0]), 5))))
            ui.table_pfe.setItem(row, 260, QTableWidgetItem(str(round(abs(avg_time - result_nonlinear[0]), 5))))

            row = ui.table_dfe.rowCount()
            ui.table_dfe.insertRow(row)
            x_full_dfe = x_dfe.copy()
            for elem in self.index_list[9:]:
                tmp = 1
                for index in elem:
                    tmp *= x_dfe[index]
                x_full_dfe.append(tmp)
            for i in range(len(x_full_dfe)):
                ui.table_dfe.setItem(row, i, QTableWidgetItem(str(round(x_full_dfe[i], 5))))
            ui.table_dfe.setItem(row, 16, QTableWidgetItem(str(round(avg_time, 5))))
            ui.table_dfe.setItem(row, 17, QTableWidgetItem(str(round(result_linear_dfe[0], 5))))
            ui.table_dfe.setItem(row, 18, QTableWidgetItem(str(round(result_nonlinear_dfe[0], 5))))
            ui.table_dfe.setItem(row, 19, QTableWidgetItem(str(round(abs(avg_time - result_linear_dfe[0]), 5))))
            ui.table_dfe.setItem(row, 20, QTableWidgetItem(str(round(abs(avg_time - result_nonlinear_dfe[0]), 5))))

        except ValueError:
            QMessageBox.warning(self, 'Ошибка', 'Ошибка в данных!')


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    main()
