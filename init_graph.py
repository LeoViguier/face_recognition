# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'init.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import sys

app = ''

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(511, 326)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.enregistrement = QtWidgets.QPushButton(self.centralwidget)
        self.enregistrement.setGeometry(QtCore.QRect(10, 230, 131, 41))
        self.enregistrement.setObjectName("enregistrement")
        self.enregistrement.clicked.connect(rec)
        self.tester = QtWidgets.QPushButton(self.centralwidget)
        self.tester.setGeometry(QtCore.QRect(190, 230, 131, 41))
        self.tester.setObjectName("tester")
        self.tester.clicked.connect(tester)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 511, 211))
        self.label.setObjectName("label")
        self.quitter = QtWidgets.QPushButton(self.centralwidget)
        self.quitter.setGeometry(QtCore.QRect(370, 230, 131, 41))
        self.quitter.setObjectName("quitter")
        self.quitter.clicked.connect(quit)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 511, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.enregistrement.setText(_translate("MainWindow", "Enregistrer"))
        self.tester.setText(_translate("MainWindow", "Tester"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Programme de reconnaissance faciale.</span></p><p align=\"center\"><br/></p><p align=\"center\"><span style=\" font-size:9pt;\">Cliquer sur Tester pour tester la reconnaissance faciale,</span></p><p align=\"center\"><span style=\" font-size:9pt;\">Cliquez sur Enregistrer pour ajouter des visages aidant à la reconnaissance.</span></p><p align=\"center\"><span style=\" font-size:9pt;\"><br/></span></p><p align=\"center\"><span style=\" font-size:9pt;\">Que voulez-vous faire?</span></p></body></html>"))
        self.quitter.setText(_translate("MainWindow", "Quitter"))


def tester():
    with open('choix.txt', 'w') as f:
        f.write('T')
    MainWindow.close()

def rec():
    with open('choix.txt', 'w') as f:
        f.write('A')
    MainWindow.close()

def quit():
    sys.exit()

def main_init_graph():
    QtCore.qInstallMessageHandler(handler)
    global MainWindow
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    app.exec_()

def handler(msg_type, msg_log_context, msg_string):
    pass