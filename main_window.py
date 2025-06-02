import sys
import ui
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QTimer
from display import Display
import time
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.gui = ui.Ui_ui()
        self.gui.setupUi(self)
        
        # 当前处理的视频
        self.current_video = None
        
        # 初始化Display类
        self.display = Display(self.gui, self)
        
        # 连接按钮信号 - 确保与Display中的方法一致
        try:
            self.gui.selectperson.clicked.disconnect()  # 先断开默认连接
            self.gui.select_video.clicked.disconnect()
            self.gui.url_video.clicked.disconnect()
            self.gui.stop_detect.clicked.disconnect()
            self.gui.clear.clicked.disconnect()
            self.gui.upload_person.clicked.disconnect()
            self.gui.fun_select.currentIndexChanged.disconnect()
            self.gui.camera_button.clicked.disconnect()  # 断开摄像头按钮连接
        except:
            # 如果没有连接过，会抛出异常，忽略它
            pass
        
        # 重新连接所有信号
        self.gui.selectperson.clicked.connect(self.display.choosePerson)
        self.gui.select_video.clicked.connect(self.display.Open)
        self.gui.url_video.clicked.connect(self.display.openUrlVideo)
        self.gui.stop_detect.clicked.connect(self.display.Close)
        self.gui.clear.clicked.connect(self.display.Clear)
        self.gui.upload_person.clicked.connect(self.display.uploadPerson)
        self.gui.fun_select.currentIndexChanged.connect(self.display.function_select)
        self.gui.camera_button.clicked.connect(self.display.openCamera)  # 连接摄像头按钮
        
        # 创建菜单
        self.setup_menu()
        
        # 状态栏更新计时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_status)
        self.timer.start(10000)  # 每10秒更新一次
        
    def setup_menu(self):
        """创建菜单和菜单项"""
        # 创建主菜单
        self.file_menu = self.menuBar().addMenu("文件")
        self.help_menu = self.menuBar().addMenu("帮助")
        
        # 创建文件菜单项
        self.open_folder_action = QtWidgets.QAction("打开视频文件夹", self)
        self.open_folder_action.triggered.connect(self.open_video_folder)
        
        self.open_url_action = QtWidgets.QAction("打开URL视频", self)
        self.open_url_action.triggered.connect(self.open_url_video)
        
        self.open_camera_action = QtWidgets.QAction("启用摄像头", self)
        self.open_camera_action.triggered.connect(self.open_camera)
        
        self.exit_action = QtWidgets.QAction("退出", self)
        self.exit_action.triggered.connect(self.close)
        
        # 添加到文件菜单
        self.file_menu.addAction(self.open_folder_action)
        self.file_menu.addAction(self.open_url_action)
        self.file_menu.addAction(self.open_camera_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.exit_action)
        
        # 创建帮助菜单项
        self.about_action = QtWidgets.QAction("关于", self)
        self.about_action.triggered.connect(self.show_about)
        self.help_menu.addAction(self.about_action)
        
    def open_video_folder(self):
        """菜单项-打开视频文件夹"""
        self.display.Open()
        
    def open_url_video(self):
        """菜单项-打开URL视频"""
        self.display.openUrlVideo()
        
    def open_camera(self):
        """菜单项-启用摄像头"""
        self.display.openCamera()
        
    def show_about(self):
        """显示关于信息"""
        QMessageBox.about(self, 
            "关于行人重识别系统", 
            "<h3>行人重识别检索系统</h3>"
            "<p>版本: 1.0.0</p>"
            "<p>本系统基于深度学习技术，实现了视频中的行人检测与重识别功能。</p>"
            "<p>支持两种行人重识别模型(MGN和BASE)，可以检索本地视频和在线视频流。</p>"
            "<p>© 2025 行人重识别项目组</p>")
            
    def update_status(self):
        """更新状态栏信息"""
        memory_info = "系统就绪"
        if hasattr(self.display, 'model') and self.display.model is not None:
            memory_info = "检测模型已加载"
        
        video_info = ""
        if hasattr(self.display, 'frameRate') and self.display.frameRate > 0:
            video_info = f" | 视频帧率: {self.display.frameRate:.1f}fps"
            
        features_info = ""
        if hasattr(self.display, 'feature_db'):
            features_info = f" | 特征库: {len(self.display.feature_db)}条记录"
            
        self.statusBar().showMessage(f"{memory_info}{video_info}{features_info}")
        
        # 更新控制面板状态标签
        if hasattr(self.display, 'targetvideo_path') and self.display.targetvideo_path:
            video_name = os.path.basename(self.display.targetvideo_path)
            self.gui.status_label.setText(f"当前视频: {video_name}")
        
    def closeEvent(self, event):
        """关闭程序时的确认"""
        reply = QMessageBox.question(self, '确认退出', 
            "确定要退出程序吗?", QMessageBox.Yes | 
            QMessageBox.No, QMessageBox.No)
            
        if reply == QMessageBox.Yes:
            # 关闭前清理资源
            if hasattr(self.display, 'stopEvent'):
                self.display.stopEvent.set()
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle("Fusion")
    
    # 创建主窗口
    mainWnd = MainWindow()
    mainWnd.show()
    
    sys.exit(app.exec_())