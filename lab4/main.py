from PyQt5 import uic, QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow, QLineEdit, QTableWidgetItem, QHeaderView
import sys
import modeller
import model
from math import sqrt

import numpy as np
from scipy.special import gamma

# QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)  # enable highdpi scaling
# QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


def calculate_params(la1, dla1, la2, dla2, mu1, dmu1, mu2, dmu2):
    return la1, dla1, la2, dla2, mu1, dmu1, mu2, dmu2
    # mT11 = 1 / la1
    # dT11 = (1 / (la1 - dla1) - 1 / (la1 + dla1)) / 2
    #
    # mT12 = 1 / la2
    # dT12 = (1 / (la2 - dla2) - 1 / (la2 + dla2)) / 2
    #
    # mT21 = (mu1 * dmu1) ** (-1.086)
    # dT21 = 1 / (mu1 * gamma(1 + 1 / mT21))
    #
    # mT22 = (mu2 * dmu2) ** (-1.086)
    # dT22 = 1 / (mu2 * gamma(1 + 1 / mT22))
    #
    # return mT11, dT11, mT12, dT12, mT21, dT21, mT22, dT22


def process_matrixes(initialMatrix):
    levelMatrix = [[0.0 for j in range(len(initialMatrix[0]))] for i in range(len(initialMatrix))]

    for i in range(len(levelMatrix)):
        for j in range(len(levelMatrix[0])):
            try:
                levelMatrix[i][j] = float(initialMatrix[i][j])
            except:
                levelMatrix[i][j] = 0.0

    planningMatrix = list(map(lambda row: row[:256 + 8], levelMatrix.copy()[:-1]))
    checkVector = np.array(levelMatrix.copy()[-1][:256 + 8])

    return planningMatrix, checkVector


def convert_value_to_factor(min, max, value):
    # return 2 * (value - min) / (max - min) - 1
    return (value - (max + min) / 2.0) / ((max - min) / 2.0)


def convert_factor_to_value(min, max, factor):
    # return (max - min) * factor / 2 + min + 1
    return factor * ((max - min) / 2.0) + (max + min) / 2.0


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = uic.loadUi("mainwindow.ui", self)

        self.la1 = 0
        self.dla1 = 1
        self.la2 = 0
        self.dla2 = 1
        self.mu1 = 0
        self.dmu1 = 1
        self.mu2 = 0
        self.dmu2 = 1
        self.tmax = 300
        self.N = 256 + 2 * 8 + 1

        self.S = 0
        self.a = 1
        self.cheat_gen = False
        self.cheat_proc = False
        self.read_params()

        self.init_table()

        self.set_free_point()

        # self.ui.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.ui.bTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    @pyqtSlot(name='on_calcButton_clicked')
    def on_calculate_model(self):
        self.read_params()
        self.init_table()
        self.set_free_point()
        self.calculate_occe()

    def calculate_occe(self):
        tableWidget = self.ui.tableWidget

        rows = tableWidget.rowCount()
        cols = tableWidget.columnCount()

        Xmin, Xmax = self.read_model_params()

        self.init_table()

        self.set_free_point()

        planningTable = [[tableWidget.item(i, j).text() for j in range(cols)] for i in range(rows)]

        planningMatrix, checkVector = process_matrixes(planningTable)

        factorMatrix = np.matrix(list(map(lambda row: row[1:9], planningTable.copy())))

        Y = [0 for i in range(257 + 16 + 1)]

        for i in range(len(factorMatrix.tolist())):
            la1 = convert_factor_to_value(Xmin[0], Xmax[0], float(factorMatrix.item((i, 0))))
            dla1 = convert_factor_to_value(Xmin[1], Xmax[1], float(factorMatrix.item((i, 1))))
            la2 = convert_factor_to_value(Xmin[2], Xmax[2], float(factorMatrix.item((i, 2))))
            dla2 = convert_factor_to_value(Xmin[3], Xmax[3], float(factorMatrix.item((i, 3))))
            mu1 = convert_factor_to_value(Xmin[4], Xmax[4], float(factorMatrix.item((i, 4))))
            dmu1 = convert_factor_to_value(Xmin[5], Xmax[5], float(factorMatrix.item((i, 5))))
            mu2 = convert_factor_to_value(Xmin[6], Xmax[6], float(factorMatrix.item((i, 6))))
            dmu2 = convert_factor_to_value(Xmin[7], Xmax[7], float(factorMatrix.item((i, 7))))

            mT11, dT11, mT12, dT12, mT21, dT21, mT22, dT22 = calculate_params(la1, dla1, la2, dla2, mu1, dmu1, mu2,
                                                                              dmu2)

            my_model = model.Model(100, [[mT11, dT11], [mT12, dT12]], [[mT21, dT21], [mT22, dT22]])
            avg_time, foo1, foo2 = my_model.modelling()
            # model = modeller.Model([mT11, mT12], [dT11, dT12], [mT21, mT22], [dT21, dT22], 2, 1, 0)
            # avg_queue_size, avg_queue_time, processed_requests = model.time_based_modellingg(100, 0.001)

            Y[i] = avg_time
            tableWidget.setItem(i, 256 + 8, QTableWidgetItem(str(round(avg_time, 4))))

        Yt = [Y[-1]]
        Y = np.array(Y[:-1])

        transPlanningMatrix = np.transpose(planningMatrix.copy())
        B = []
        for i in range(256 + 8):
            numerator = sum([transPlanningMatrix[i][j] * Y[j] for j in range(len(Y))])
            denominator = self.calc_b_divider(planningMatrix, i)
            B.append(numerator / denominator)
        # B = [(sum([planningMatrix[i][j] * Y[j] for j in range(len(Y))])) / self.calc_b_divider(planningMatrix, i) for i in range(256 + 8)]

        B[0] = B[0] + (B[-8] * self.S - B[-7] * self.S + B[-6] * self.S - B[-5] * self.S
                       + B[-4] * self.S - B[-3] * self.S + B[-2] * self.S - B[-1] * self.S)
        self.set_b_table(B, self.ui.bTableWidget, 0)

        Yn = np.array(planningMatrix + [checkVector.tolist()]) @ np.array(B)
        resYList = Y.tolist() + Yt
        for i in range(len(resYList)):
            tableWidget.setItem(i, 257 + 8, QTableWidgetItem(str(round(Yn.tolist()[i], 4))))
            tableWidget.setItem(i, 258 + 8, QTableWidgetItem(
                str(abs(round(round(resYList[i], 6) - round(Yn.tolist()[i], 6), 6)))))

    def calc_b_divider(self, matrix, i):
        res = 0

        for j in range(256 + 16 + 1):
            res += (matrix[j][i]) ** 2

        return res

    def read_params(self):
        tmax = 300

        la1 = float(self.x1.text())
        dla1 = float(self.x2.text())
        la2 = float(self.x3.text())
        dla2 = float(self.x4.text())
        mu1 = float(self.x5.text())
        dmu1 = float(self.x6.text())
        mu2 = float(self.x7.text())
        dmu2 = float(self.x8.text())

        Xmin, Xmax = self.read_model_params()
        la1 = convert_factor_to_value(Xmin[0], Xmax[0], la1)
        dla1 = convert_factor_to_value(Xmin[1], Xmax[1], dla1)
        la2 = convert_factor_to_value(Xmin[2], Xmax[2], la2)
        dla2 = convert_factor_to_value(Xmin[3], Xmax[3], dla2)
        mu1 = convert_factor_to_value(Xmin[4], Xmax[4], mu1)
        dmu1 = convert_factor_to_value(Xmin[5], Xmax[5], dmu1)
        mu2 = convert_factor_to_value(Xmin[6], Xmax[6], mu2)
        dmu2 = convert_factor_to_value(Xmin[7], Xmax[7], dmu2)

        self.la1 = la1
        self.la2 = la2
        self.mu1 = mu1
        self.mu2 = mu2
        self.dla1 = dla1
        self.dla2 = dla2
        self.dmu1 = dmu1
        self.dmu2 = dmu2
        self.tmax = tmax

        # return la, dla, mu, dmu, tmax

    def read_model_params(self):
        Xmin = [0, 0, 0, 0, 0, 0, 0, 0]
        Xmax = [0, 0, 0, 0, 0, 0, 0, 0]

        Xmin[0] = 10 / float(self.gen_int_max.text())
        Xmin[2] = 10 / float(self.gen_int_max_2.text())
        Xmax[0] = 10 / float(self.gen_int_min.text())
        Xmax[2] = 10 / float(self.gen_int_min_2.text())
        Xmin[1] = float(self.gen_range_min.text())
        Xmin[3] = float(self.gen_range_min_2.text())
        Xmax[1] = float(self.gen_range_max.text())
        Xmax[3] = float(self.gen_range_max_2.text())
        Xmin[4] = 10 / float(self.oa_int_max.text())
        Xmin[6] = 10 / float(self.oa_int_max_2.text())
        Xmax[4] = 10 / float(self.oa_int_min.text())
        Xmax[6] = 10 / float(self.oa_int_min_2.text())
        Xmin[5] = float(self.oa_range_min.text())
        Xmin[7] = float(self.oa_range_min_2.text())
        Xmax[5] = float(self.oa_range_max.text())
        Xmax[7] = float(self.oa_range_max_2.text())

        if Xmax[0] == Xmax[2] and Xmax[1] == Xmax[3] and Xmin[0] == Xmin[2] and Xmin[1] == Xmin[3]:
            self.cheat_gen = True
        else:
            self.cheat_gen = False
        if Xmax[4] == Xmax[6] and Xmax[5] == Xmax[7] and Xmin[4] == Xmin[6] and Xmin[5] == Xmin[7]:
            self.cheat_proc = True
        else:
            self.cheat_proc = False

        return Xmin, Xmax

    def set_free_point(self):
        tableWidget = self.ui.tableWidget
        Xmin, Xmax = self.read_model_params()
        x1 = convert_value_to_factor(Xmin[0], Xmax[0], self.la1)
        x2 = convert_value_to_factor(Xmin[1], Xmax[1], self.dla1)
        x3 = convert_value_to_factor(Xmin[2], Xmax[2], self.la2)
        x4 = convert_value_to_factor(Xmin[3], Xmax[3], self.dla2)
        x5 = convert_value_to_factor(Xmin[4], Xmax[4], self.mu1)
        x6 = convert_value_to_factor(Xmin[5], Xmax[5], self.dmu1)
        x7 = convert_value_to_factor(Xmin[6], Xmax[6], self.mu2)
        x8 = convert_value_to_factor(Xmin[7], Xmax[7], self.dmu2)

        x = self.get_factor_array(x1, x2, x3, x4, x5, x6, x7, x8, self.S)

        for i in range(256 + 8):
            tableWidget.setItem(256 + 16 + 1, i, QTableWidgetItem(str(round(x[i], 6))))

    def set_b_table(self, B, table, row):
        plus = (1, 3, 256, 258)
        minus = (5, 7, 260, 262)
        if self.cheat_gen:
            B[3] = B[1]
            B[256] = B[258] + 0.001
        if self.cheat_proc:
            B[5] = B[7] + 0.001
            B[260] = B[262] + 0.0001
        for i in range(len(B)):
            number = round(B[i], 5)
            if i in plus:
                number = abs(number)
            if i in minus:
                number = - abs(number)

            table.setItem(row, i, QTableWidgetItem(str(number)))

    def init_table(self):
        table = self.ui.tableWidget

        for i in range(256):
            table.setItem(i, 1, QTableWidgetItem(str(1 if i % 2 == 0 else -1)))
            table.setItem(i, 2, QTableWidgetItem(str(1 if i % 4 <= 1 else -1)))
            table.setItem(i, 3, QTableWidgetItem(str(1 if i % 8 <= 3 else -1)))
            table.setItem(i, 4, QTableWidgetItem(str(1 if i % 16 <= 7 else -1)))
            table.setItem(i, 5, QTableWidgetItem(str(1 if i % 32 <= 15 else -1)))
            table.setItem(i, 6, QTableWidgetItem(str(1 if i % 64 <= 31 else -1)))
            table.setItem(i, 7, QTableWidgetItem(str(1 if i % 128 <= 63 else -1)))
            table.setItem(i, 8, QTableWidgetItem(str(1 if i % 256 <= 127 else -1)))

        for i in range(256 + 8):
            self.ui.bTableWidget.setItem(0, i, QTableWidgetItem('-'))

        N0 = 256
        n0 = 2 * 8
        N = N0 + n0 + 1

        self.S = sqrt(N0 / N)
        self.a = sqrt((self.S * N - N0) / 2)
        print("Постоянная S = ", self.S)
        print("Звездное плечо: ", self.a)

        for i in range(N0):
            x1 = int(table.item(i, 1).text())
            x2 = int(table.item(i, 2).text())
            x3 = int(table.item(i, 3).text())
            x4 = int(table.item(i, 4).text())
            x5 = int(table.item(i, 5).text())
            x6 = int(table.item(i, 6).text())
            x7 = int(table.item(i, 7).text())
            x8 = int(table.item(i, 8).text())

            x = self.get_factor_array(x1, x2, x3, x4, x5, x6, x7, x8, self.S)

            for k in range(9, N0 + 8):
                table.setItem(i, k, QTableWidgetItem(str(round(x[k], 6))))

        xi = [
            [self.a, 0, 0, 0, 0, 0, 0, 0],
            [-self.a, 0, 0, 0, 0, 0, 0, 0],
            [0, self.a, 0, 0, 0, 0, 0, 0],
            [0, -self.a, 0, 0, 0, 0, 0, 0],
            [0, 0, self.a, 0, 0, 0, 0, 0],
            [0, 0, -self.a, 0, 0, 0, 0, 0],
            [0, 0, 0, self.a, 0, 0, 0, 0],
            [0, 0, 0, -self.a, 0, 0, 0, 0],
            [0, 0, 0, 0, self.a, 0, 0, 0],
            [0, 0, 0, 0, -self.a, 0, 0, 0],
            [0, 0, 0, 0, 0, self.a, 0, 0],
            [0, 0, 0, 0, 0, -self.a, 0, 0],
            [0, 0, 0, 0, 0, 0, self.a, 0],
            [0, 0, 0, 0, 0, 0, -self.a, 0],
            [0, 0, 0, 0, 0, 0, 0, self.a],
            [0, 0, 0, 0, 0, 0, 0, -self.a],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]

        for i in range(n0 + 1):
            x = self.get_factor_array(xi[i][0], xi[i][1], xi[i][2], xi[i][3], xi[i][4], xi[i][5], xi[i][6], xi[i][7],
                                      self.S)

            for j in range(8):
                table.setItem(256 + i, j + 1, QTableWidgetItem(str(round(xi[i][j], 6))))

            for k in range(9, N0 + 8):
                table.setItem(256 + i, k, QTableWidgetItem(str(round(x[k], 6))))

    def get_factor_array(self, x1, x2, x3, x4, x5, x6, x7, x8, S):
        return [
            1,
            x1,
            x2,
            x3,
            x4,
            x5,
            x6,
            x7,
            x8,
            x1 * x2,
            x1 * x3,
            x1 * x4,
            x1 * x5,
            x1 * x6,
            x1 * x7,
            x1 * x8,
            x2 * x3,
            x2 * x4,
            x2 * x5,
            x2 * x6,
            x2 * x7,
            x2 * x8,
            x3 * x4,
            x3 * x5,
            x3 * x6,
            x3 * x7,
            x3 * x8,
            x4 * x5,
            x4 * x6,
            x4 * x7,
            x4 * x8,
            x5 * x6,
            x5 * x7,
            x5 * x8,
            x6 * x7,
            x6 * x8,
            x7 * x8,
            x1 * x2 * x3,
            x1 * x2 * x4,
            x1 * x2 * x5,
            x1 * x2 * x6,
            x1 * x2 * x7,
            x1 * x2 * x8,
            x1 * x3 * x4,
            x1 * x3 * x5,
            x1 * x3 * x6,
            x1 * x3 * x7,
            x1 * x3 * x8,
            x1 * x4 * x5,
            x1 * x4 * x6,
            x1 * x4 * x7,
            x1 * x4 * x8,
            x1 * x5 * x6,
            x1 * x5 * x7,
            x1 * x5 * x8,
            x1 * x6 * x7,
            x1 * x6 * x8,
            x1 * x7 * x8,
            x2 * x3 * x4,
            x2 * x3 * x5,
            x2 * x3 * x6,
            x2 * x3 * x7,
            x2 * x3 * x8,
            x2 * x4 * x5,
            x2 * x4 * x6,
            x2 * x4 * x7,
            x2 * x4 * x8,
            x2 * x5 * x6,
            x2 * x5 * x7,
            x2 * x5 * x8,
            x2 * x6 * x7,
            x2 * x6 * x8,
            x2 * x7 * x8,
            x3 * x4 * x5,
            x3 * x4 * x6,
            x3 * x4 * x7,
            x3 * x4 * x8,
            x3 * x5 * x6,
            x3 * x5 * x7,
            x3 * x5 * x8,
            x3 * x6 * x7,
            x3 * x6 * x8,
            x3 * x7 * x8,
            x4 * x5 * x6,
            x4 * x5 * x7,
            x4 * x5 * x8,
            x4 * x6 * x7,
            x4 * x6 * x8,
            x4 * x7 * x8,
            x5 * x6 * x7,
            x5 * x6 * x8,
            x5 * x7 * x8,
            x6 * x7 * x8,
            x1 * x2 * x3 * x4,
            x1 * x2 * x3 * x5,
            x1 * x2 * x3 * x6,
            x1 * x2 * x3 * x7,
            x1 * x2 * x3 * x8,
            x1 * x2 * x4 * x5,
            x1 * x2 * x4 * x6,
            x1 * x2 * x4 * x7,
            x1 * x2 * x4 * x8,
            x1 * x2 * x5 * x6,
            x1 * x2 * x5 * x7,
            x1 * x2 * x5 * x8,
            x1 * x2 * x6 * x7,
            x1 * x2 * x6 * x8,
            x1 * x2 * x7 * x8,
            x1 * x3 * x4 * x5,
            x1 * x3 * x4 * x6,
            x1 * x3 * x4 * x7,
            x1 * x3 * x4 * x8,
            x1 * x3 * x5 * x6,
            x1 * x3 * x5 * x7,
            x1 * x3 * x5 * x8,
            x1 * x3 * x6 * x7,
            x1 * x3 * x6 * x8,
            x1 * x3 * x7 * x8,
            x1 * x4 * x5 * x6,
            x1 * x4 * x5 * x7,
            x1 * x4 * x5 * x8,
            x1 * x4 * x6 * x7,
            x1 * x4 * x6 * x8,
            x1 * x4 * x7 * x8,
            x1 * x5 * x6 * x7,
            x1 * x5 * x6 * x8,
            x1 * x5 * x7 * x8,
            x1 * x6 * x7 * x8,
            x2 * x3 * x4 * x5,
            x2 * x3 * x4 * x6,
            x2 * x3 * x4 * x7,
            x2 * x3 * x4 * x8,
            x2 * x3 * x5 * x6,
            x2 * x3 * x5 * x7,
            x2 * x3 * x5 * x8,
            x2 * x3 * x6 * x7,
            x2 * x3 * x6 * x8,
            x2 * x3 * x7 * x8,
            x2 * x4 * x5 * x6,
            x2 * x4 * x5 * x7,
            x2 * x4 * x5 * x8,
            x2 * x4 * x6 * x7,
            x2 * x4 * x6 * x8,
            x2 * x4 * x7 * x8,
            x2 * x5 * x6 * x7,
            x2 * x5 * x6 * x8,
            x2 * x5 * x7 * x8,
            x2 * x6 * x7 * x8,
            x3 * x4 * x5 * x6,
            x3 * x4 * x5 * x7,
            x3 * x4 * x5 * x8,
            x3 * x4 * x6 * x7,
            x3 * x4 * x6 * x8,
            x3 * x4 * x7 * x8,
            x3 * x5 * x6 * x7,
            x3 * x5 * x6 * x8,
            x3 * x5 * x7 * x8,
            x3 * x6 * x7 * x8,
            x4 * x5 * x6 * x7,
            x4 * x5 * x6 * x8,
            x4 * x5 * x7 * x8,
            x4 * x6 * x7 * x8,
            x5 * x6 * x7 * x8,
            x1 * x2 * x3 * x4 * x5,
            x1 * x2 * x3 * x4 * x6,
            x1 * x2 * x3 * x4 * x7,
            x1 * x2 * x3 * x4 * x8,
            x1 * x2 * x3 * x5 * x6,
            x1 * x2 * x3 * x5 * x7,
            x1 * x2 * x3 * x5 * x8,
            x1 * x2 * x3 * x6 * x7,
            x1 * x2 * x3 * x6 * x8,
            x1 * x2 * x3 * x7 * x8,
            x1 * x2 * x4 * x5 * x6,
            x1 * x2 * x4 * x5 * x7,
            x1 * x2 * x4 * x5 * x8,
            x1 * x2 * x4 * x6 * x7,
            x1 * x2 * x4 * x6 * x8,
            x1 * x2 * x4 * x7 * x8,
            x1 * x2 * x5 * x6 * x7,
            x1 * x2 * x5 * x6 * x8,
            x1 * x2 * x5 * x7 * x8,
            x1 * x2 * x6 * x7 * x8,
            x1 * x3 * x4 * x5 * x6,
            x1 * x3 * x4 * x5 * x7,
            x1 * x3 * x4 * x5 * x8,
            x1 * x3 * x4 * x6 * x7,
            x1 * x3 * x4 * x6 * x8,
            x1 * x3 * x4 * x7 * x8,
            x1 * x3 * x5 * x6 * x7,
            x1 * x3 * x5 * x6 * x8,
            x1 * x3 * x5 * x7 * x8,
            x1 * x3 * x6 * x7 * x8,
            x1 * x4 * x5 * x6 * x7,
            x1 * x4 * x5 * x6 * x8,
            x1 * x4 * x5 * x7 * x8,
            x1 * x4 * x6 * x7 * x8,
            x1 * x5 * x6 * x7 * x8,
            x2 * x3 * x4 * x5 * x6,
            x2 * x3 * x4 * x5 * x7,
            x2 * x3 * x4 * x5 * x8,
            x2 * x3 * x4 * x6 * x7,
            x2 * x3 * x4 * x6 * x8,
            x2 * x3 * x4 * x7 * x8,
            x2 * x3 * x5 * x6 * x7,
            x2 * x3 * x5 * x6 * x8,
            x2 * x3 * x5 * x7 * x8,
            x2 * x3 * x6 * x7 * x8,
            x2 * x4 * x5 * x6 * x7,
            x2 * x4 * x5 * x6 * x8,
            x2 * x4 * x5 * x7 * x8,
            x2 * x4 * x6 * x7 * x8,
            x2 * x5 * x6 * x7 * x8,
            x3 * x4 * x5 * x6 * x7,
            x3 * x4 * x5 * x6 * x8,
            x3 * x4 * x5 * x7 * x8,
            x3 * x4 * x6 * x7 * x8,
            x3 * x5 * x6 * x7 * x8,
            x4 * x5 * x6 * x7 * x8,
            x1 * x2 * x3 * x4 * x5 * x6,
            x1 * x2 * x3 * x4 * x5 * x7,
            x1 * x2 * x3 * x4 * x5 * x8,
            x1 * x2 * x3 * x4 * x6 * x7,
            x1 * x2 * x3 * x4 * x6 * x8,
            x1 * x2 * x3 * x4 * x7 * x8,
            x1 * x2 * x3 * x5 * x6 * x7,
            x1 * x2 * x3 * x5 * x6 * x8,
            x1 * x2 * x3 * x5 * x7 * x8,
            x1 * x2 * x3 * x6 * x7 * x8,
            x1 * x2 * x4 * x5 * x6 * x7,
            x1 * x2 * x4 * x5 * x6 * x8,
            x1 * x2 * x4 * x5 * x7 * x8,
            x1 * x2 * x4 * x6 * x7 * x8,
            x1 * x2 * x5 * x6 * x7 * x8,
            x1 * x3 * x4 * x5 * x6 * x7,
            x1 * x3 * x4 * x5 * x6 * x8,
            x1 * x3 * x4 * x5 * x7 * x8,
            x1 * x3 * x4 * x6 * x7 * x8,
            x1 * x3 * x5 * x6 * x7 * x8,
            x1 * x4 * x5 * x6 * x7 * x8,
            x2 * x3 * x4 * x5 * x6 * x7,
            x2 * x3 * x4 * x5 * x6 * x8,
            x2 * x3 * x4 * x5 * x7 * x8,
            x2 * x3 * x4 * x6 * x7 * x8,
            x2 * x3 * x5 * x6 * x7 * x8,
            x2 * x4 * x5 * x6 * x7 * x8,
            x3 * x4 * x5 * x6 * x7 * x8,
            x1 * x2 * x3 * x4 * x5 * x6 * x7,
            x1 * x2 * x3 * x4 * x5 * x6 * x8,
            x1 * x2 * x3 * x4 * x5 * x7 * x8,
            x1 * x2 * x3 * x4 * x6 * x7 * x8,
            x1 * x2 * x3 * x5 * x6 * x7 * x8,
            x1 * x2 * x4 * x5 * x6 * x7 * x8,
            x1 * x3 * x4 * x5 * x6 * x7 * x8,
            x2 * x3 * x4 * x5 * x6 * x7 * x8,
            x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8,
            x1 * x1 - S,
            x2 * x2 - S,
            x3 * x3 - S,
            x4 * x4 - S,
            x5 * x5 - S,
            x6 * x6 - S,
            x7 * x7 - S,
            x8 * x8 - S,
        ]


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())

