# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mine_matching.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap, QImage
from matching import matching_demo
import os
import cv2

img_path= ''
img2_path = ''

class Ui_MainWindow(object):
    def pushButtonClicked(self):
        self.label_4.setText("Waiting")
        self.verticalSlider.setSliderPosition(50)
        self.label_5.clear()
        self.label_6.clear()
        self.label_7.clear()
        self.label_8.clear()
        fname = QtWidgets.QFileDialog.getOpenFileName()
        global img_path
        img_path = fname[0]
        img = QPixmap(fname[0])
        img = img.scaled(self.label_5.width(),self.label_5.height())
        self.label_5.setPixmap(img)

    def pushButton2Clicked(self):
        fname2 = QtWidgets.QFileDialog.getOpenFileName()
        global img2_path
        img2_path = fname2[0]
        img2 = QPixmap(fname2[0])
        img2 = img2.scaled(self.label_6.width(),self.label_6.height())
        self.label_6.setPixmap(img2)

    def pushButton3Clicked(self):
        self.label_4.setText("Proceeding...")
        img3 = demo(img_path, img2_path) # 1080,1080, 3
        self.label_4.setText("Done...")
        imge = QPixmap(os.getcwd()+'/result.jpg')

        imge = imge.scaled(self.label_7.width(),self.label_7.height())
        self.label_7.setPixmap(imge)
        tar = cv2.imread(img2_path)
        result = cv2.imread(os.getcwd() + '/result.jpg')
        alpha = 0.5
        beta = (1.0 - alpha)
        shrink = cv2.resize(tar, None, fx=result.shape[0] / tar.shape[0], fy=result.shape[0] / tar.shape[0],
                            interpolation=cv2.INTER_AREA)
        dst = cv2.addWeighted(shrink, alpha, result, beta, 0.0, None)
        cv2.imwrite('overlay.jpg', dst)
        imge2 = QPixmap(os.getcwd()+'/overlay.jpg')
        imge2 = imge2.scaled(self.label_8.width(),self.label_8.height())
        self.label_8.setPixmap(imge2)

    def detectChange(self):
        tar = cv2.imread(img2_path)
        result = cv2.imread(os.getcwd() + '/result.jpg')
        alpha = 0.01 * self.verticalSlider.value()
        beta = (1.0 - alpha)
        shrink = cv2.resize(tar, None, fx=result.shape[0] / tar.shape[0], fy=result.shape[0] / tar.shape[0],
                            interpolation=cv2.INTER_AREA)
        dst = cv2.addWeighted(shrink, alpha, result, beta, 0.0, None)
        cv2.imwrite('overlay.jpg', dst)
        imge2 = QPixmap(os.getcwd()+'/overlay.jpg')
        imge2 = imge2.scaled(self.label_8.width(),self.label_8.height())
        self.label_8.setPixmap(imge2)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(731, 921)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(10, 70, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.pushButtonClicked)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 0, 331, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(210, 70, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.pushButton2Clicked)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 120, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.pushButton3Clicked)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(410, 20, 121, 121))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        addlogo = QPixmap(os.getcwd()+'/ADD_Logo.jpg')
        logo1 = addlogo.scaled(self.label_2.width(),self.label_2.height())
        self.label_2.setPixmap(logo1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(560, 20, 121, 121))
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        kulogo = QPixmap(os.getcwd()+'/KU_Logo.jpg')
        logo2 = kulogo.scaled(self.label_3.width(),self.label_3.height())
        self.label_3.setPixmap(logo2)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(210, 110, 191, 41))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(10, 10, 171, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.label_4.setFont(font)
        self.label_4.setTextFormat(QtCore.Qt.AutoText)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(10, 160, 337, 337))
        self.label_5.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(360, 160, 337, 337))
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(10, 520, 337, 337))
        self.label_7.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(360, 520, 337, 337))
        self.label_8.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(150, 500, 111, 16))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(500, 500, 111, 16))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(470, 860, 111, 16))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(140, 860, 111, 16))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.verticalSlider = QtWidgets.QSlider(self.centralwidget)
        self.verticalSlider.setGeometry(QtCore.QRect(700, 520, 22, 341))
        self.verticalSlider.setMaximum(100)
        self.verticalSlider.setSliderPosition(50)
        self.verticalSlider.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider.setInvertedAppearance(False)
        self.verticalSlider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.verticalSlider.setTickInterval(10)
        self.verticalSlider.setObjectName("verticalSlider")
        self.verticalSlider.valueChanged.connect(self.detectChange)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 731, 21))
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
        self.pushButton.setText(_translate("MainWindow", "Loading source image")) # 입력영상 불러오기
        self.label.setText(_translate("MainWindow", "Aerial image matching program")) # 딥러닝 기반 항공 영상 정합 프로그램
        self.pushButton_2.setText(_translate("MainWindow", "Loading target image")) # 기준영상 불러오기
        self.pushButton_3.setText(_translate("MainWindow", "Executing image matching")) # 영상 정합 수행
        self.groupBox.setTitle(_translate("MainWindow", "Current progress")) # 정합 진행 상태
        self.label_4.setText(_translate("MainWindow", "Waiting")) # 대기중
        self.label_9.setText(_translate("MainWindow", "Source image")) # 입력영상
        self.label_10.setText(_translate("MainWindow", "Target image")) # 기준영상
        self.label_11.setText(_translate("MainWindow", "Overlay check")) # 정합 결과 중첩
        self.label_12.setText(_translate("MainWindow", "Matching result")) # 정합 결과

if __name__ == "__main__":
    import sys
    demo = matching_demo()
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

