from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
import sys
from model import Model, intensity_to_param, get_plot


class MainWindow(QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = uic.loadUi("gui.ui", self)
        self.ui.generator_radio_m.toggled.connect(self.radio1_clicked)
        self.ui.generator_radio_lambda.toggled.connect(self.radio2_clicked)
        self.radio = 1

    def radio1_clicked(self, enabled):
        if enabled:
            self.radio = 1
            self.ui.lineEdit_generator_lambda.setEnabled(False)
            self.ui.lineEdit_generator_m.setEnabled(True)
            self.ui.lineEdit_processor_lambda.setEnabled(False)
            self.ui.lineEdit_processor_m.setEnabled(True)

    def radio2_clicked(self, enabled):
        if enabled:
            self.radio = 2
            self.ui.lineEdit_generator_lambda.setEnabled(True)
            self.ui.lineEdit_generator_m.setEnabled(False)
            self.ui.lineEdit_processor_lambda.setEnabled(True)
            self.ui.lineEdit_processor_m.setEnabled(False)

    @pyqtSlot(name='on_pushButton_clicked')
    def _parse_parameters(self):
        try:
            ui = self.ui
            if self.radio == 1:
                m1 = float(ui.lineEdit_generator_m.text())
                m2 = float(ui.lineEdit_processor_m.text())
                lambda1 = 10 / m1
                lambda2 = 10 / m2
            else:
                lambda1 = float(ui.lineEdit_generator_lambda.text())
                lambda2 = float(ui.lineEdit_processor_lambda.text())
                m1 = intensity_to_param(lambda1)
                m2 = intensity_to_param(lambda2)
            d1 = float(ui.lineEdit_generator_sigma.text())
            d2 = float(ui.lineEdit_processor_sigma.text())
            duration = int(ui.lineEdit_duration.text())
            my_model = Model(duration, [[m1, d1]], [[m2, d2]])
            avg_time, processed, generated = my_model.modelling()
            self._show_results(avg_time, processed, lambda1, lambda2, generated)
            get_plot(lambda1, d1)

        except ValueError:
            QMessageBox.warning(self, 'Ошибка', 'Ошибка в данных!')

    def _show_results(self, results_time, results_processed, lambda1, lambda2, result_generated):
        ui = self.ui
        ui.lineEdit_res_processed.setText(str(results_processed))
        ui.lineEdit_res_avg_time.setText(str(round(results_time, 4)))
        ro = round(lambda1 / lambda2, 4)
        ro_fact = round(result_generated / results_processed, 4)
        ui.lineEdit_ro.setText(str(ro))
        ui.lineEdit_ro_fact.setText(str(ro_fact))


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    main()
