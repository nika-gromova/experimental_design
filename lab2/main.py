from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QTableWidgetItem, QMessageBox, QMainWindow
import sys
from model import Model, intensity_to_param, get_avg_model
from experiment import *


def generate_linear_str(coefficients):
    result_str = ""
    for i in range(len(coefficients)):
        sign = " - " if coefficients[i] < 0 else " + "
        result_str += sign + str(round(abs(coefficients[i]), 4)) + "x" + str(i)
    return result_str


def generate_nonlinear_str(coefficients):
    result_str = ""
    combs = get_comb([1, 2, 3, 4])
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
        self.ui = uic.loadUi("gui2.ui", self)
        self.params = []
        # кол-во заявок
        self.count = 500
        self.vars = 4
        self.plan = get_plan_matrix(self.vars)
        self.plan_full = get_all_x(self.plan)
        self.ui.table.setHorizontalHeaderLabels(["x0", "x1", "x2", "x3", "x4", "x1x2", "x1x3", "x1x4", "x2x3", "x2x4",
                                                 "x3x4", "x1x2x3", "x1x2x4", "x1x3x4", "x2x3x4", "x1x2x3x4",
                                                 "y", "yл", "yчнл", "|y-yл|", "|y-yчнл|"])
        self.coefficients = []

    def calculate_y(self):
        plan = self.plan
        y = []
        for row in range(len(plan)):

            m1 = self.params[0][0 if plan[row][1] == -1 else 1]
            d1 = self.params[1][0 if plan[row][2] == -1 else 1]
            m2 = self.params[2][0 if plan[row][3] == -1 else 1]
            d2 = self.params[3][0 if plan[row][4] == -1 else 1]
            my_model = Model(self.count, [[m1, d1]], [[m2, d2]])
            avg_time, foo1, foo2 = my_model.modelling()
            y.append(avg_time)
        return y

    def calculate_main(self):
        y = self.calculate_y()
        coefficients = calculate_coefficients(self.plan, y)
        linear = calculate(coefficients, self.plan)
        nonlinear = calculate_nonlinear(coefficients, self.plan)
        variance_linear = [abs(y[i] - linear[i]) for i in range(len(y))]
        variance_nonlinear = [abs(y[i] - nonlinear[i]) for i in range(len(y))]

        self.coefficients = coefficients
        return [y, coefficients, linear, nonlinear, variance_linear, variance_nonlinear]

    @pyqtSlot(name='on_calculate_clicked')
    def _parse_parameters(self):
        try:
            ui = self.ui
            lambda1_min = intensity_to_param(float(ui.min_lambda_gen.text()))
            lambda1_max = intensity_to_param(float(ui.max_lambda_gen.text()))
            lambda2_min = intensity_to_param(float(ui.min_lambda_proc.text()))
            lambda2_max = intensity_to_param(float(ui.max_lambda_proc.text()))

            sigma1_min = float(ui.min_sigma_gen.text())
            sigma1_max = float(ui.max_sigma_gen.text())
            sigma2_min = float(ui.min_sigma_proc.text())
            sigma2_max = float(ui.max_sigma_proc.text())

            self.params.append([lambda1_min, lambda1_max])
            self.params.append([sigma1_min, sigma1_max])
            self.params.append([lambda2_min, lambda2_max])
            self.params.append([sigma2_min, sigma2_max])

            results = self.calculate_main()
            self._show_results(results)

            # вставить в таблицу self.plan_full

            for row in range(len(self.plan_full)):
                for column in range(len(self.plan_full[0])):
                    item = QTableWidgetItem(str(self.plan_full[row][column]))
                    ui.table.setItem(row, column, item)
                ui.table.setItem(row, 16, QTableWidgetItem(str(round(results[0][row], 5))))
                ui.table.setItem(row, 17, QTableWidgetItem(str(round(results[2][row], 5))))
                ui.table.setItem(row, 18, QTableWidgetItem(str(round(results[3][row], 5))))
                ui.table.setItem(row, 19, QTableWidgetItem(str(round(results[4][row], 5))))
                ui.table.setItem(row, 20, QTableWidgetItem(str(round(results[5][row], 5))))

        except ValueError:
            QMessageBox.warning(self, 'Ошибка', 'Ошибка в данных!')

    def _show_results(self, results):
        ui = self.ui
        str_linear = generate_linear_str(results[1][:5])
        str_nonlinear = generate_nonlinear_str(results[1])
        ui.linear_text.setText(str_linear)
        ui.nonlinear_text.setText(str_nonlinear)

    @pyqtSlot(name='on_custom_clicked')
    def custom_dot(self):
        try:
            norm = (-1, 0, 1)

            ui = self.ui

            x1 = int(ui.x1.text())
            x2 = int(ui.x2.text())
            x3 = int(ui.x3.text())
            x4 = int(ui.x4.text())
            x = [1, x1, x2, x3, x4]
            for elem in x:
                if elem not in norm:
                    QMessageBox.warning(self, 'Ошибка', 'Ошибка в данных!')

            m1 = self.params[0][0 if x1 == -1 else 1]
            d1 = self.params[1][0 if x2 == -1 else 1]
            m2 = self.params[2][0 if x3 == -1 else 1]
            d2 = self.params[3][0 if x3 == -1 else 1]
            my_model = Model(self.count, [[m1, d1]], [[m2, d2]])
            avg_time = get_avg_model(my_model, 10)

            result_linear = calculate(self.coefficients, [x])
            result_nonlinear = calculate_nonlinear(self.coefficients, [x])

            # insert row и вставить новые данные
            row = ui.table.rowCount()
            ui.table.insertRow(row)
            x_full = get_x_row_nonlinear(x)
            for i in range(len(x_full)):
                ui.table.setItem(row, i, QTableWidgetItem(str(round(x_full[i], 5))))
            ui.table.setItem(row, 16, QTableWidgetItem(str(round(avg_time, 5))))
            ui.table.setItem(row, 17, QTableWidgetItem(str(round(result_linear[0], 5))))
            ui.table.setItem(row, 18, QTableWidgetItem(str(round(result_nonlinear[0], 5))))
            ui.table.setItem(row, 19, QTableWidgetItem(str(round(abs(avg_time - result_linear[0]), 5))))
            ui.table.setItem(row, 20, QTableWidgetItem(str(round(abs(avg_time - result_nonlinear[0]), 5))))

        except ValueError:
            QMessageBox.warning(self, 'Ошибка', 'Ошибка в данных!')


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    main()
