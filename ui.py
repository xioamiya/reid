# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QPixmap


class Ui_ui(object):
    def setupUi(self, ui):
        ui.setObjectName("ui")
        ui.resize(1200, 900)
        ui.setWindowIcon(QIcon("icon.png") if QPixmap("icon.png").width() > 0 else QIcon())
        ui.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #333333;
                font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
            }
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border-radius: 4px;
                padding: 5px;
                font-weight: bold;
                font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
            QPushButton:pressed {
                background-color: #2a66c8;
            }
            QComboBox {
                border: 1px solid #bbbbbb;
                border-radius: 3px;
                padding: 1px 18px 1px 3px;
                background-color: white;
            }
            QScrollArea {
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
            QFrame {
                border-radius: 5px;
            }
        """)
        
        self.centralwidget = QtWidgets.QWidget(ui)
        self.centralwidget.setObjectName("centralwidget")
        
        # 设置左侧面板
        self.left_panel = QtWidgets.QFrame(self.centralwidget)
        self.left_panel.setGeometry(QtCore.QRect(10, 10, 240, 440))
        self.left_panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.left_panel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.left_panel.setStyleSheet("background-color: #ffffff; border: 1px solid #dddddd;")
        self.left_panel.setObjectName("left_panel")
        
        # 模型选择
        self.model_label = QtWidgets.QLabel(self.left_panel)
        self.model_label.setGeometry(QtCore.QRect(20, 15, 200, 25))
        self.model_label.setAlignment(QtCore.Qt.AlignCenter)
        self.model_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4a86e8;")
        self.model_label.setObjectName("model_label")
        
        self.fun_select = QtWidgets.QComboBox(self.left_panel)
        self.fun_select.setGeometry(QtCore.QRect(60, 45, 120, 32))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setWeight(75)
        self.fun_select.setFont(font)
        self.fun_select.setObjectName("fun_select")
        self.fun_select.addItem("")
        self.fun_select.addItem("")
        self.fun_select.setEditable(True)
        self.ledit = self.fun_select.lineEdit()
        self.ledit.setAlignment(Qt.AlignCenter)
        self.fun_select.model().item(0).setTextAlignment(Qt.AlignCenter)
        self.fun_select.model().item(1).setTextAlignment(Qt.AlignCenter)
        
        # 目标人员显示
        self.match_result_label = QtWidgets.QLabel(self.left_panel)
        self.match_result_label.setGeometry(QtCore.QRect(20, 85, 200, 25))
        self.match_result_label.setAlignment(QtCore.Qt.AlignCenter)
        self.match_result_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4a86e8;")
        self.match_result_label.setObjectName("match_result_label")
        
        # 添加人像框边框
        self.query_person_frame = QtWidgets.QFrame(self.left_panel)
        self.query_person_frame.setGeometry(QtCore.QRect(46, 115, 148, 256))
        self.query_person_frame.setStyleSheet("background-color: #fafafa; border: 2px solid #4a86e8; border-radius: 5px;")
        
        self.query_person = QtWidgets.QLabel(self.left_panel)
        self.query_person.setGeometry(QtCore.QRect(56, 120, 128, 246))
        self.query_person.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.query_person.setAlignment(QtCore.Qt.AlignCenter)
        self.query_person.setStyleSheet("border: none; background-color: transparent;")
        self.query_person.setObjectName("query_person")
        
        # 选择人员按钮样式设置
        person_button_style = """
            QPushButton {
                font-size: 13px;
                height: 36px;
                background-color: #0f9d58;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                text-align: center;
                padding-left: 10px;
                padding-right: 10px;
            }
            QPushButton:hover {
                background-color: #0b8043;
            }
            QPushButton:pressed {
                background-color: #096536;
            }
        """
        
        self.selectperson = QtWidgets.QPushButton(self.left_panel)
        self.selectperson.setGeometry(QtCore.QRect(20, 380, 200, 36))
        self.selectperson.setObjectName("selectperson")
        self.selectperson.setStyleSheet(person_button_style)
        self.selectperson.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        
        self.upload_person = QtWidgets.QPushButton(self.left_panel)
        self.upload_person.setGeometry(QtCore.QRect(20, 415, 200, 36))
        self.upload_person.setObjectName("upload_person")
        self.upload_person.setStyleSheet(person_button_style)
        self.upload_person.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        
        # 视频显示区
        self.video_frame = QtWidgets.QFrame(self.centralwidget)
        self.video_frame.setGeometry(QtCore.QRect(260, 10, 930, 545))
        self.video_frame.setStyleSheet("background-color: #000000; border-radius: 8px;")
        
        self.target_video = QtWidgets.QLabel(self.centralwidget)
        self.target_video.setGeometry(QtCore.QRect(265, 15, 920, 535))
        self.target_video.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.target_video.setAlignment(QtCore.Qt.AlignCenter)
        self.target_video.setStyleSheet("color: #ffffff; font-size: 16px;")
        self.target_video.setObjectName("target_video")
        
        # 按钮控制区
        self.control_panel = QtWidgets.QFrame(self.centralwidget)
        self.control_panel.setGeometry(QtCore.QRect(260, 560, 930, 50))
        self.control_panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.control_panel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.control_panel.setStyleSheet("background-color: #ffffff; border: 1px solid #dddddd;")
        self.control_panel.setObjectName("control_panel")
        
        # 控制按钮样式
        control_button_style = """
            QPushButton {
                font-size: 13px;
                height: 36px;
                background-color: #4285f4;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                text-align: center;
                padding-left: 10px;
                padding-right: 10px;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
            QPushButton:pressed {
                background-color: #2a66c8;
            }
        """
        
        self.select_video = QtWidgets.QPushButton(self.control_panel)
        self.select_video.setGeometry(QtCore.QRect(15, 7, 150, 36))
        self.select_video.setObjectName("select_video")
        self.select_video.setStyleSheet(control_button_style)
        self.select_video.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        
        self.url_video = QtWidgets.QPushButton(self.control_panel)
        self.url_video.setGeometry(QtCore.QRect(175, 7, 150, 36))
        self.url_video.setObjectName("url_video")
        self.url_video.setStyleSheet(control_button_style)
        self.url_video.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        
        self.camera_button = QtWidgets.QPushButton(self.control_panel)
        self.camera_button.setGeometry(QtCore.QRect(335, 7, 150, 36))
        self.camera_button.setObjectName("camera_button")
        self.camera_button.setStyleSheet(control_button_style.replace("#4285f4", "#9370db"))
        self.camera_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        
        self.stop_detect = QtWidgets.QPushButton(self.control_panel)
        self.stop_detect.setGeometry(QtCore.QRect(495, 7, 150, 36))
        self.stop_detect.setObjectName("stop_detect")
        self.stop_detect.setStyleSheet(control_button_style.replace("#4285f4", "#db4437"))
        self.stop_detect.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        
        self.clear = QtWidgets.QPushButton(self.control_panel)
        self.clear.setGeometry(QtCore.QRect(655, 7, 150, 36))
        self.clear.setObjectName("clear")
        self.clear.setStyleSheet(control_button_style.replace("#4285f4", "#f4b400"))
        self.clear.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        
        # 添加状态信息标签
        self.status_label = QtWidgets.QLabel(self.control_panel)
        self.status_label.setGeometry(QtCore.QRect(815, 7, 265, 36))
        self.status_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.status_label.setStyleSheet("color: #666666; font-size: 12px; padding-right: 10px;")
        self.status_label.setText("准备就绪")
        
        # 匹配结果显示区
        self.match_display = QtWidgets.QScrollArea(self.centralwidget)
        self.match_display.setGeometry(QtCore.QRect(10, 460, 240, 418))
        self.match_display.setStyleSheet("background-color: #ffffff; border: 1px solid #dddddd; border-radius: 5px;")
        self.match_display.setWidgetResizable(True)
        self.match_display.setObjectName("match_display")
        match_content = QtWidgets.QWidget()
        self.match_display.setWidget(match_content)
        
        # 识别结果展示区
        self.result_frame = QtWidgets.QFrame(self.centralwidget)
        self.result_frame.setGeometry(QtCore.QRect(260, 615, 930, 263))
        self.result_frame.setStyleSheet("background-color: #ffffff; border: 1px solid #dddddd;")
        
        self.result_label = QtWidgets.QLabel(self.centralwidget)
        self.result_label.setGeometry(QtCore.QRect(260, 615, 930, 30))
        self.result_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4a86e8; background-color: #f5f5f5; border-bottom: 1px solid #dddddd;")
        self.result_label.setObjectName("result_label")
        
        # 添加匹配信息显示面板
        self.match_info_panel = QtWidgets.QFrame(self.centralwidget)
        self.match_info_panel.setGeometry(QtCore.QRect(10, 880, 1180, 40))
        self.match_info_panel.setStyleSheet("background-color: #f0f0f0; border: 1px solid #dddddd; border-radius: 5px;")
        self.match_info_panel.setObjectName("match_info_panel")
        
        self.match_info_label = QtWidgets.QLabel(self.match_info_panel)
        self.match_info_label.setGeometry(QtCore.QRect(10, 5, 1160, 30))
        self.match_info_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.match_info_label.setStyleSheet("font-size: 13px; color: #333333; font-weight: bold; background-color: transparent; border: none;")
        self.match_info_label.setObjectName("match_info_label")
        self.match_info_label.setText("等待检测结果...")
        
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(260, 645, 930, 233))
        self.scrollArea.setStyleSheet("background-color: #ffffff; border: none;")
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 928, 231))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.layout = QGridLayout(self.scrollAreaWidgetContents)
        self.layout.setSpacing(10)
        
        # 设置主布局
        main_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        ui.setCentralWidget(self.centralwidget)
        
        # 菜单栏和状态栏
        self.menubar = QtWidgets.QMenuBar(ui)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 26))
        self.menubar.setObjectName("menubar")
        ui.setMenuBar(self.menubar)
        
        self.statusbar = QtWidgets.QStatusBar(ui)
        self.statusbar.setObjectName("statusbar")
        self.statusbar.setStyleSheet("color: #666666;")
        ui.setStatusBar(self.statusbar)
        self.statusbar.showMessage("视频行人重识别系统已启动")

        self.retranslateUi(ui)
        QtCore.QMetaObject.connectSlotsByName(ui)

    def retranslateUi(self, ui):
        _translate = QtCore.QCoreApplication.translate
        ui.setWindowTitle(_translate("ui", "视频行人检索软件系统"))
        self.model_label.setText(_translate("ui", "选择重识别模型"))
        self.match_result_label.setText(_translate("ui", "特征匹配结果"))
        self.query_person.setText(_translate("ui", "目标人员"))
        self.target_video.setText(_translate("ui", "待检测视频"))
        self.selectperson.setText(_translate("ui", "选择目标人员"))
        self.upload_person.setText(_translate("ui", "上传目标人员到特征库"))
        self.select_video.setText(_translate("ui", "打开视频文件夹"))
        self.url_video.setText(_translate("ui", "打开URL视频"))
        self.camera_button.setText(_translate("ui", "启用摄像头"))
        self.stop_detect.setText(_translate("ui", "停止检测"))
        self.clear.setText(_translate("ui", "清除结果列表"))
        self.result_label.setText(_translate("ui", "检索结果"))
        self.match_info_label.setText(_translate("ui", "等待检测结果..."))
        self.fun_select.setItemText(0, _translate("ui", "MGN"))
        self.fun_select.setItemText(1, _translate("ui", "BASE"))