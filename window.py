
import sys
import cv2
import os
import math
import numpy as np
import torch
from torchvision.transforms import transforms
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# 导入 MGN ReID 模型相关
from MGN.model import Model as reidModel
from MGN.option import args
import MGN.utils.utility as utility

# 导入 YOLOv5
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import (check_img_size, non_max_suppression, scale_coords)
from yolov5.utils.datasets import letterbox
from yolov5.utils.torch_utils import time_synchronized
from torch.nn import functional as F


class VideoStreamThread(QThread):
    update_frame = pyqtSignal(QImage)

    def __init__(self, source, q_feature, device, yolo_model, reid_model, img_transform, thresh):
        super().__init__()
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        self.q_feature = q_feature
        self.device = device
        self.model = yolo_model
        self.model_ReID = reid_model
        self.transform = img_transform
        self.thres = thresh
        self.half = device.type != 'cpu'
        self.imgsz = check_img_size(640, s=int(self.model.stride.max()))
        self.stride = int(self.model.stride.max())
        self.conf_thres = 0.3
        self.iou_thres = 0.5
        self.classes = 0

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            # YOLOv5 推理
            t0 = time_synchronized()
            img = letterbox(frame, self.imgsz, stride=self.stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = self.model(img, augment=False)[0]
            detections = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes)[0]
            t1 = time_synchronized()
            # 处理检测结果
            if detections is not None and len(detections):
                # Rescale boxes to original frame
                detections[:, :4] = scale_coords(img.shape[2:], detections[:, :4], frame.shape).round()
                for *xyxy, conf, cls in detections.cpu().numpy():
                    x1, y1, x2, y2 = map(int, xyxy)
                    # 只处理行人类别
                    if int(cls) != 0:
                        continue
                    person = frame[y1:y2, x1:x2]
                    if person.size == 0:
                        continue
                    # ReID 特征提取
                    pil_img = Image.fromarray(cv2.cvtColor(person, cv2.COLOR_BGR2RGB))
                    with torch.no_grad():
                        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                        feat = self.model_ReID(tensor)[0].cpu()
                    # 相似度计算
                    sim = F.cosine_similarity(F.normalize(self.q_feature), F.normalize(feat), dim=1)[0].item()
                    # 阈值判断
                    color = (0, 255, 0) if sim > self.thres else (0, 0, 255)
                    label = f"{sim*100:.1f}%"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # 转换为 QImage 并更新 UI
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.update_frame.emit(q_img)
        self.cap.release()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("实时行人检索系统")
        self.setGeometry(100, 100, 1024, 768)

        # 预加载模型
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        # ReID 模型
        ckpt = utility.checkpoint(args)
        ckpt.dir = "./MGN/output/mgn/"
        self.model_ReID = reidModel(args, ckpt).to(self.device)
        # YOLOv5 模型
        self.yolo_model = attempt_load("./yolov5/weights/yolov5x.pt", map_location=self.device)
        if self.device.type != 'cpu':
            self.yolo_model.half()
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.q_feature = None
        self.thres = 0.87
        self.stream_thread = None

        # 构建 UI
        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # 目标选择
        btn_select = QPushButton("选择目标图片")
        btn_select.clicked.connect(self.choose_person)
        layout.addWidget(btn_select)

        # 视频源选择
        self.source_selector = QComboBox()
        self.source_selector.addItem("选择视频源")
        self.source_selector.addItem("摄像头")
        self.source_selector.addItem("RTSP 流")
        layout.addWidget(self.source_selector)

        # 开始 / 停止
        self.btn_start = QPushButton("开始检索")
        self.btn_start.clicked.connect(self.start_stream)
        layout.addWidget(self.btn_start)
        self.btn_stop = QPushButton("停止检索")
        self.btn_stop.clicked.connect(self.stop_stream)
        self.btn_stop.setEnabled(False)
        layout.addWidget(self.btn_stop)

        # 视频显示
        self.video_label = QLabel()
        self.video_label.setFixedSize(960, 540)
        layout.addWidget(self.video_label)

    def choose_person(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择目标人员图片', '', 'Images (*.jpg *.png *.bmp)')
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGB")
        with torch.no_grad():
            tensor = self.transform(pil).unsqueeze(0).to(self.device)
            feat = self.model_ReID(tensor)[0].cpu()
        self.q_feature = feat
        QMessageBox.information(self, "提示", "目标特征提取完成！")

    def start_stream(self):
        if self.q_feature is None:
            QMessageBox.warning(self, "警告", "请先选择目标图片！")
            return
        idx = self.source_selector.currentIndex()
        if idx == 1:
            source = 0
        elif idx == 2:
            url, ok = QInputDialog.getText(self, '输入 RTSP URL', 'RTSP 流地址:')
            if not ok or not url:
                return
            source = url
        else:
            QMessageBox.warning(self, "警告", "请选择视频源！")
            return
        # 启动线程
        self.stream_thread = VideoStreamThread(
            source, self.q_feature, self.device,
            self.yolo_model, self.model_ReID,
            self.transform, self.thres
        )
        self.stream_thread.update_frame.connect(self.video_label.setPixmap)
        self.stream_thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_stream(self):
        if self.stream_thread:
            self.stream_thread.terminate()
            self.stream_thread.wait()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def closeEvent(self, event):
        self.stop_stream()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

