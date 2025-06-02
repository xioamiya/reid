from PyQt5.QtWidgets import QProgressBar, QLabel, QWidget, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer
import time

class ProgressBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.start_time = None
        self.total_frames = 0
        self.current_frame = 0
        
    def init_ui(self):
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat('%p%')
        layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel('准备就绪')
        layout.addWidget(self.status_label)
        
        # 剩余时间标签
        self.time_label = QLabel('剩余时间: --:--')
        layout.addWidget(self.time_label)
        
        # 更新定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        
    def start(self, total_frames):
        """开始进度跟踪"""
        self.total_frames = total_frames
        self.current_frame = 0
        self.start_time = time.time()
        self.progress_bar.setValue(0)
        self.status_label.setText('正在处理...')
        self.timer.start(1000)  # 每秒更新一次
        
    def update(self, frame_num):
        """更新进度"""
        self.current_frame = frame_num
        progress = int((frame_num / self.total_frames) * 100)
        self.progress_bar.setValue(progress)
        
    def update_time(self):
        """更新剩余时间估计"""
        if self.start_time is None or self.current_frame == 0:
            return
            
        elapsed_time = time.time() - self.start_time
        frames_per_second = self.current_frame / elapsed_time
        remaining_frames = self.total_frames - self.current_frame
        
        if frames_per_second > 0:
            remaining_seconds = remaining_frames / frames_per_second
            minutes = int(remaining_seconds // 60)
            seconds = int(remaining_seconds % 60)
            self.time_label.setText(f'剩余时间: {minutes:02d}:{seconds:02d}')
        
    def finish(self):
        """完成进度跟踪"""
        self.timer.stop()
        self.progress_bar.setValue(100)
        self.status_label.setText('处理完成')
        self.time_label.setText('剩余时间: 00:00')
        self.start_time = None
        
    def reset(self):
        """重置进度条"""
        self.timer.stop()
        self.progress_bar.setValue(0)
        self.status_label.setText('准备就绪')
        self.time_label.setText('剩余时间: --:--')
        self.start_time = None
        self.total_frames = 0
        self.current_frame = 0