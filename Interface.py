import sys
import LB
from PyQt5.QtWidgets import (QWidget, QPushButton,QLineEdit,QFileDialog,
                             QHBoxLayout, QVBoxLayout, QApplication)


class Feather_opener(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.input = QLineEdit(self)
        self.open = QPushButton("Open")
        self.choose = QPushButton("Choose")
        self.open.clicked.connect(self.open_clicked)
        self.choose.clicked.connect(self.choose_clicked)

        hbox = QHBoxLayout()
        hbox.addWidget(self.input)
        hbox.addWidget(self.open)
        hbox.addWidget(self.choose)
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)

        self.setLayout(vbox)
        self.show()

    def choose_clicked(self):
        input_text = self.input.text()
        if input_text:
            path, format_egal = QFileDialog.getOpenFileName(self, 'Input Dialog', input_text, "*.feather")
        else:
            path, format_egal = QFileDialog.getOpenFileName(self, 'Input Dialog', 'D:\Stock\Market\CN', "*.feather")

        if path not in [None,""]:
            self.input.setText(path)
            print("choose",path)
            LB.feather_csv_converter(path)

    def open_clicked(self):
        path = self.input.text()
        if path not in [None, ""]:
            print("open", path)
            LB.feather_csv_converter(path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Feather_opener()
    sys.exit(app.exec_())