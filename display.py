import cv2
import os
import math
import numpy as np
import torch
from MGN.model import Model as reidModel
from MGN.option import args
import MGN.utils.utility as utility
from torchvision.transforms import transforms
from PIL import ImageFont, Image, ImageDraw
import threading
from PyQt5.QtWidgets import QFileDialog, QLabel, QVBoxLayout, QWidget, QInputDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from torch.nn import functional as F
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams, letterbox
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_synchronized
import pickle
import glob
import shutil
import time


class Display:
    def __init__(self, ui, mainWnd):
        self.ui = ui
        self.ui.stop_detect.setEnabled(False)
        self.mainWnd = mainWnd
        
        # 尝试加载支持中文的字体
        try:
            self.font = ImageFont.truetype("simhei.ttf", 20, 0)  # 使用黑体
        except:
            try:
                self.font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 20, 0)
            except:
                try:
                    self.font = ImageFont.truetype("font/simhei.ttf", 20, 0)
                except:
                    self.font = ImageFont.truetype("font/platech.ttf", 20, 0)
                    print("警告: 未找到中文字体，使用默认字体")
                    
        self.netMain = None
        self.metaMain = None
        self.altNames = None
        self.count = 0
        self.queryimg_path = ''
        self.targetvideo_path = ''
        self.img_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.thres = 0.87
        
        # 采用懒加载模式，只在需要时加载模型
        self.model_ReID = None  # 延迟加载
        self.model = None       # 延迟加载
        
        # 创建存放匹配结果的Widget
        self.match_widget = QWidget()
        self.match_layout = QVBoxLayout(self.match_widget)
        self.ui.match_display.setWidget(self.match_widget)

        self.ui.selectperson.clicked.connect(self.choosePerson)
        self.ui.upload_person.clicked.connect(self.uploadPerson)
        self.ui.select_video.clicked.connect(self.Open)
        self.ui.stop_detect.clicked.connect(self.Close)
        self.ui.clear.clicked.connect(self.Clear)
        self.ui.fun_select.currentIndexChanged.connect(self.function_select)
        self.ui.url_video.clicked.connect(self.openUrlVideo)
        self.ui.camera_button.clicked.connect(self.openCamera)
        self.stopEvent = threading.Event()
        self.stopEvent.clear()
        
        # 加载特征数据库
        self.feature_db = {}
        self.load_feature_database()

    def load_feature_database(self):
        """加载特征数据库"""
        try:
            # 尝试加载合并后的特征库
            feature_db_path = os.path.join("person_db_features", "market1501_combined_features.pkl")
            if os.path.exists(feature_db_path):
                with open(feature_db_path, 'rb') as f:
                    self.feature_db = pickle.load(f)
                print(f"成功加载特征库，包含 {len(self.feature_db)} 个不同ID的行人特征")
                
                # 统计信息
                total_images = sum(len(features) for features in self.feature_db.values())
                print(f"特征库中共有 {total_images} 张图像的特征")
                
                # 更新状态信息
                self.update_status(f"特征库加载完成: {len(self.feature_db)}个ID, {total_images}张图像")
                
                # 只保留相对路径
                for pid, feats in self.feature_db.items():
                    for feat in feats:
                        feat['img_path'] = os.path.join('person_db_features', 'images', os.path.basename(feat['img_path']))
                
                # 保存更新后的特征库
                with open(feature_db_path, 'wb') as f:
                    pickle.dump(self.feature_db, f)
                
                return
                
            # 如果没有找到合并特征库，尝试加载单独的特征库
            feature_db_path = os.path.join("person_db_features", "market1501_features.pkl")
            if os.path.exists(feature_db_path):
                with open(feature_db_path, 'rb') as f:
                    self.feature_db = pickle.load(f)
                print(f"成功加载特征库，包含 {len(self.feature_db)} 个不同ID的行人特征")
                self.update_status(f"特征库加载完成: {len(self.feature_db)}个ID")
                return
                
            print("未找到特征库文件，创建空特征库")
            self.feature_db = {}
            os.makedirs("person_db_features", exist_ok=True)
            
        except Exception as e:
            print(f"加载特征数据库失败: {e}")
            self.feature_db = {}
            self.update_status("特征库加载失败")

    def uploadPerson(self):
        """上传目标人员到特征库"""
        if not self.queryimg_path:
            QMessageBox.warning(self.mainWnd, "提示", "请先选择目标人员图片")
            return
            
        # 获取人员ID
        person_id, ok = QInputDialog.getText(self.mainWnd, "输入人员ID", "请输入人员ID (仅包含字母、数字和下划线):")
        if not ok or not person_id:
            return
            
        # 验证ID格式
        import re
        if not re.match(r'^[a-zA-Z0-9_]+$', person_id):
            QMessageBox.warning(self.mainWnd, "错误", "人员ID只能包含字母、数字和下划线")
            return
            
        try:
            # 确保特征已提取
            if not hasattr(self, 'q_feature'):
                QMessageBox.warning(self.mainWnd, "错误", "未能提取特征，请重新选择图片")
                return
                
            # 创建特征数据
            img_name = os.path.basename(self.queryimg_path)
            new_img_path = os.path.join('person_db_features', 'images', img_name)
            feature_data = {
                'feature': self.q_feature,
                'img_path': new_img_path,
                'person_id': person_id,
                'camera_id': 'upload',
                'frame_id': '0',
                'dataset_type': 'custom'
            }
            
            # 更新特征库
            if person_id not in self.feature_db:
                self.feature_db[person_id] = []
            self.feature_db[person_id].append(feature_data)
            
            # 保存更新后的特征库
            feature_db_path = os.path.join("person_db_features", "market1501_combined_features.pkl")
            with open(feature_db_path, 'wb') as f:
                pickle.dump(self.feature_db, f)
                
            QMessageBox.information(self.mainWnd, "成功", f"已成功将人员 {person_id} 添加到特征库")
            self.update_status(f"特征库已更新: {len(self.feature_db)}个ID")
            
        except Exception as e:
            QMessageBox.critical(self.mainWnd, "错误", f"保存特征失败: {str(e)}")

    def match_with_database(self, query_feature):
        """与特征数据库中的特征匹配，返回top3结果"""
        results = []
        
        # 遍历特征库中的每个ID
        for person_id, feature_list in self.feature_db.items():
            # 对每个ID的所有特征计算相似度
            for feature_data in feature_list:
                try:
                    # 获取特征向量
                    feature = feature_data['feature']
                    img_path = feature_data['img_path']
                    
                    # 计算相似度
                    similarity = self.Align(query_feature, feature)
                    similarity_value = similarity.item() if hasattr(similarity, 'item') else float(similarity)
                    
                    # 添加结果，包含更多信息
                    results.append({
                        'person_id': person_id,
                        'img_path': img_path,
                        'similarity': similarity_value,
                        'camera_id': feature_data.get('camera_id', 'unknown'),
                        'frame_id': feature_data.get('frame_id', 'unknown'),
                        'dataset_type': feature_data.get('dataset_type', 'unknown')
                    })
                    
                except Exception as e:
                    print(f"计算与 {person_id} 的相似度时出错: {e}")
        
        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 返回top3结果
        return results[:3]

    def display_top_matches(self, top_matches):
        """显示top3匹配结果"""
        # 清除之前的匹配结果
        for i in reversed(range(self.match_layout.count())): 
            self.match_layout.itemAt(i).widget().deleteLater()
        
        if not top_matches:
            no_result = QLabel("没有找到匹配结果")
            no_result.setStyleSheet("color: #666666; font-size: 14px;")
            self.match_layout.addWidget(no_result)
            return
            
        # 显示新的匹配结果
        for i, match in enumerate(top_matches):
            # 创建结果容器
            result_container = QWidget()
            result_layout = QVBoxLayout(result_container)
            result_layout.setSpacing(5)
            
            # 创建标签显示ID和相似度
            id_label = QLabel(f"ID: {match['person_id']}")
            id_label.setStyleSheet("color: #4a86e8; font-weight: bold; font-size: 14px;")
            
            sim_label = QLabel(f"相似度: {match['similarity']*100:.2f}%")
            sim_label.setStyleSheet("color: #0f9d58; font-size: 13px;")
            
            # 显示图片
            img_label = QLabel()
            img_label.setFixedSize(128, 256)
            
            img_path = match['img_path']
            # 修改：使用相对于reid_system目录的路径
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            img_path_full = os.path.join(base_dir, 'reid_system', img_path)
            print("尝试加载图片路径：", img_path_full, "存在吗？", os.path.exists(img_path_full))

            if os.path.exists(img_path_full):
                try:
                    img = cv2.imdecode(np.fromfile(img_path_full, dtype=np.uint8), -1)
                    if img is None:
                        img_label.setText("图片解码失败")
                    else:
                        img = cv2.resize(img, (128, 256))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        h, w, c = img.shape
                        qimg = QImage(img, w, h, 3*w, QImage.Format_RGB888)
                        img_label.setPixmap(QPixmap.fromImage(qimg))
                except Exception as e:
                    img_label.setText(f"图片加载失败: {e}")
            else:
                img_label.setText(f"图片不存在: {img_path_full}")
            
            # 添加详细信息
            info_label = QLabel(f"摄像头: {match['camera_id']}\n帧号: {match['frame_id']}\n类型: {match['dataset_type']}")
            info_label.setStyleSheet("color: #666666; font-size: 12px;")
            
            # 将所有元素添加到结果容器
            result_layout.addWidget(id_label)
            result_layout.addWidget(sim_label)
            result_layout.addWidget(img_label)
            result_layout.addWidget(info_label)
            
            # 添加结果容器到主布局
            self.match_layout.addWidget(result_container)
            
            # 设置结果容器的样式
            result_container.setStyleSheet("""
                QWidget {
                    background-color: #ffffff;
                    border: 1px solid #dddddd;
                    border-radius: 5px;
                    padding: 5px;
                }
            """)

    def function_select(self):
        self.function_choose = self.ui.fun_select.currentText()
        if self.function_choose == "MGN":
            print("切换模型为MGN")
            self.ensure_reid_model_loaded()
            self.thres = 0.87
        else:
            print("切换模型为BASE")
            self.ensure_reid_model_loaded()
            self.thres = 0.825

    def openUrlVideo(self):
        """打开URL视频流"""
        from PyQt5.QtWidgets import QInputDialog, QMessageBox
        url, ok = QInputDialog.getText(self.mainWnd, '输入视频URL', '请输入视频URL地址或预设关键字(cctv1/test1/demo):')
        if not ok or not url:
            return
            
        # 使用get_stream_url获取直接可用的流地址
        stream_url = get_stream_url(url)
        if not stream_url:
            QMessageBox.warning(self.mainWnd, "错误", "无法解析此类型的视频URL")
            return
            
        # 验证URL是否可访问
        try:
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                QMessageBox.warning(self.mainWnd, "错误", "无法打开视频URL，请检查地址是否正确")
                return
            
            print(f"成功打开视频流: {stream_url}")
            self.targetvideo_path = stream_url
            self.cap = cap
            
            # 获取帧率，如果获取失败或过低，则设置一个合理的默认值
            self.frameRate = self.cap.get(cv2.CAP_PROP_FPS)
            if self.frameRate <= 0 or self.frameRate > 100:
                self.frameRate = 25  # 设置默认帧率为25fps
            
            # 设置缓冲区大小，提高流畅度
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            
            print(f"视频帧率: {self.frameRate}fps")
            
            # 询问是否要调整处理帧率
            fps_adjust, ok = QInputDialog.getInt(
                self.mainWnd, "调整处理帧率", 
                "请输入处理帧率 (1-30，越低检测越密集但可能更慢):", 
                min=1, max=30, value=5
            )
            if ok:
                self.process_frameRate = fps_adjust
            else:
                self.process_frameRate = 5  # 默认每5帧处理一次
            
            # 启动视频处理线程
            self.ui.select_video.setEnabled(False)
            self.ui.url_video.setEnabled(False)
            self.ui.stop_detect.setEnabled(True)
            th = threading.Thread(target=self.process_video)
            th.start()
        except Exception as e:
            QMessageBox.warning(self.mainWnd, "错误", f"打开URL视频失败: {str(e)}")

    def Open(self):
        """打开视频文件夹进行检索"""
        self.directory = QFileDialog.getExistingDirectory(None, '选择视频文件夹', './')
        if self.directory:
            self.update_status(f"正在处理文件夹: {os.path.basename(self.directory)}")
            self.Display()
        else:
            print("未选择文件夹")
            self.update_status("未选择文件夹")

    def process_video(self):
        """处理单个视频流"""
        self.file_handle = open('out_trajectory.txt', mode='w')
        currframenum = 1
        
        # 设置每隔多少帧检测一次
        process_interval = self.process_frameRate if hasattr(self, 'process_frameRate') else 5
        
        # 更新状态信息
        self.update_status("正在处理视频...")
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                # 只处理帧间隔为process_interval的帧，其他帧只显示不处理
                if currframenum % process_interval == 0:
                    # 处理帧并检测人员
                    processed_frame = self.YOLO(frame, currframenum, os.path.basename(self.targetvideo_path), self.frameRate)
                    processed_frame = cv2.resize(processed_frame, (1000, 563))
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                    h, w, c = processed_frame.shape
                    img = QImage(processed_frame, w, h, 3 * w, QImage.Format_RGB888)
                    self.ui.target_video.setPixmap(QPixmap.fromImage(img))
                else:
                    # 只显示，不处理
                    frame = cv2.resize(frame, (1000, 563))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, c = frame.shape
                    img = QImage(frame, w, h, 3 * w, QImage.Format_RGB888)
                    self.ui.target_video.setPixmap(QPixmap.fromImage(img))
                
                # 控制刷新速度，防止UI卡顿
                wait_time = int(1000 / (self.frameRate * 1.5))  # 加快1.5倍处理速度
                cv2.waitKey(max(1, wait_time))
                
                if True == self.stopEvent.is_set():
                    self.stopEvent.clear()
                    self.ui.target_video.clear()
                    self.ui.stop_detect.setEnabled(False)
                    self.ui.select_video.setEnabled(True)
                    self.ui.url_video.setEnabled(True)
                    self.update_status("视频处理已停止")
                    break
            else:
                # 对于网络流，可能会有临时断连，尝试重新连接
                if self.targetvideo_path.startswith(("http", "rtsp", "rtmp")):
                    print("视频流中断，尝试重新连接...")
                    self.update_status("视频流中断，正在重新连接...")
                    self.cap.release()
                    time.sleep(2)  # 等待2秒再重试
                    self.cap = cv2.VideoCapture(self.targetvideo_path)
                    if not self.cap.isOpened():
                        print("重新连接失败，退出播放")
                        self.update_status("重新连接失败")
                        break
                    continue
                else:
                    break
            currframenum += 1
            
        self.cap.release()
        self.file_handle.close()
        self.ui.target_video.clear()
        self.ui.stop_detect.setEnabled(False)
        self.ui.select_video.setEnabled(True)
        self.ui.url_video.setEnabled(True)
        self.update_status("视频处理完成")

    def Close(self):
        self.stopEvent.set()
        # self.ui.select_video.setEnabled(True)
        # self.ui.stop_detect.setEnabled(False)

    def Clear(self):
        """清除匹配结果"""
        for i in range(self.ui.layout.count()):
            self.ui.layout.itemAt(i).widget().deleteLater()
        
        # 重置计数器
        self.count = 0
        
        # 更新匹配信息面板
        if hasattr(self.ui, 'match_info_label'):
            self.ui.match_info_label.setText("结果已清除，等待下一次检测...")
            self.ui.match_info_label.setStyleSheet("font-size: 13px; color: #333333; font-weight: bold; background-color: transparent; border: none;")
        
        # 更新结果标签
        self.ui.result_label.setText("检索结果")
        self.ui.result_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4a86e8; background-color: #f5f5f5; border-bottom: 1px solid #dddddd;")
        
        # 更新状态信息
        self.update_status("已清除所有匹配结果")

    def choosePerson(self):
        # 打开一张图，opencv读取，提取特征q_feature
        self.queryimg_path, self.fileType = QFileDialog.getOpenFileName(self.mainWnd, '选择图片', '')
        if self.queryimg_path != '':
            # img=cv2.imread(self.queryimg_path)
            img = cv2.imdecode(np.fromfile(self.queryimg_path, dtype=np.uint8), -1)
            person_img = Image.fromarray(img).convert("RGB")
            self.q_feature = self.Extract(person_img)
            img = cv2.resize(img, (128, 256))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            h, w, c = img.shape
            img = QImage(img, w, h, 3 * w, QImage.Format_RGB888)
            self.ui.query_person.setPixmap(QPixmap.fromImage(img))
            
            # 与特征库匹配并显示top3结果
            top_matches = self.match_with_database(self.q_feature)
            self.display_top_matches(top_matches)

    def Display(self):
        if len(self.directory) > 0:
            self.file_handle = open('out_trajectory.txt', mode='w')
            self.ui.select_video.setEnabled(False)
            self.ui.url_video.setEnabled(False)
            self.ui.stop_detect.setEnabled(True)
            # 针对打开的文件夹，得到所有的视频，作为检索库
            filelist = os.listdir(self.directory)
            # 遍历视频
            for i in range(len(filelist)):
                filename = os.path.join(self.directory, filelist[i])
                cap = cv2.VideoCapture(filename)
                frameRate = math.ceil(cap.get(cv2.CAP_PROP_FPS))
                currframenum = 1
                while cap.isOpened():
                    # 读取第i个图片
                    success, frame = cap.read()
                    if success:
                        if currframenum % (2*frameRate) != 0:
                            currframenum += 1
                            continue
                        currframenum += 1
                        # 通过yolo进行目标检测
                        frame = self.YOLO(frame,currframenum,filelist[i],frameRate)
                        frame = cv2.resize(frame, (1000, 563))
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        h, w, c = frame.shape
                        img = QImage(frame, w, h, 3 * w, QImage.Format_RGB888)
                        # 得到检索的图片
                        self.ui.target_video.setPixmap(QPixmap.fromImage(img))
                        cv2.waitKey(int(1000 / frameRate))
                        if True == self.stopEvent.is_set():
                            self.stopEvent.clear()
                            self.ui.target_video.clear()
                            self.ui.stop_detect.setEnabled(False)
                            self.ui.select_video.setEnabled(True)
                            self.ui.url_video.setEnabled(True)
                            return
                    else:
                        break
                cap.release()
            # self.Close()
            # self.stopEvent.clear()
            self.ui.target_video.clear()
            self.ui.stop_detect.setEnabled(False)
            self.ui.select_video.setEnabled(True)
            self.ui.url_video.setEnabled(True)
            self.file_handle.close()

            # currframenum = 1
        # while self.cap.isOpened():
            # success, frame = self.cap.read()
            # if success:
                # if currframenum % (math.ceil(self.frameRate)) != 0:
                    # currframenum += 1
                    # continue
                # frame = self.YOLO(frame,currframenum)
                # frame = cv2.resize(frame, (800, 450))
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # h, w, c = frame.shape
                # img = QImage(frame, w, h, 3 * w, QImage.Format_RGB888)
                # self.ui.target_video.setPixmap(QPixmap.fromImage(img))
                # cv2.waitKey(int(1000 / math.ceil(self.frameRate)))
                # if True == self.stopEvent.is_set():
                    # self.stopEvent.clear()
                    # self.ui.target_video.clear()
                    # self.ui.stop_detect.setEnabled(False)
                    # self.ui.select_video.setEnabled(True)
                    # break
            # else:
                # break
            # currframenum += 1
        # self.ui.target_video.setPixmap(QPixmap(""))

    def YOLO(self,frame,currframenum,filename,frameRate):
        # 确保YOLO模型已加载
        self.ensure_yolo_model_loaded()
        
        img = letterbox(frame, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=False)[0]
        # Apply NMS
        imghw = [img.shape[2], img.shape[3]]
        detections = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=False)
        t2 = time_synchronized()
        detections = detections[0].cpu().numpy()
        # yolov5前向推断得到检测结果
        person_boxs, person_class_names = self.preprocess(detections,frame.shape[0],frame.shape[1], imghw)
        # 遍历每一个检测结果
        for box in person_boxs:
            x1 = int(box[0])
            x2 = int(box[0] + box[2])
            y1 = int(box[1])
            y2 = int(box[1] + box[3])
            if x1 < 0:
                x1 = 0
            if x2 > frame.shape[1]:
                x2 = frame.shape[1]
            if y1 < 0:
                y1 = 0
            if y2 > frame.shape[0]:
                y2 = frame.shape[0]
            # 把行人在图片中的区域扣取出来
            person = frame[y1:y2, x1:x2]
            person_img = Image.fromarray(person)
            # 提取扣取出来的区域的特征t_feature
            t_feature = self.Extract(person_img)
            # 计算扣取出来的区域的特征t_feature和要查询的人的图片的特征q_feature之间的距离
            distance = self.Align(self.q_feature,t_feature)
            # 设定阈值，>0.7则为同一个人
            if distance > self.thres:
                similaritylabel = QLabel()
                similaritylabel.setText("相似度: %.2f%%"%((distance)*100))
                similaritylabel.move(100 * self.count + 2, 510)
                targetlabel = QLabel()
                targetlabel.setFixedSize(128, 256)
                person = cv2.resize(person, (128, 256))
                person = cv2.cvtColor(person, cv2.COLOR_RGB2BGR)
                h, w, c = person.shape
                # 把检索到相同身份的这个人保存下来
                targetperson = QImage(person, w, h, 3 * w, QImage.Format_RGB888)
                targetlabel.setPixmap(QPixmap.fromImage(targetperson))
                targetlabel.move(100 * self.count + 2, 530)
                timelabel = QLabel()
                hour, minute, second = self.getTime(currframenum,frameRate)
                timelabel.setText('出现时间：'+str(hour)+':'+str(minute)+':'+str(second))
                timelabel.move(100 * self.count + 2, 786)
                videolabel = QLabel()
                videosource = '视频源:' + filename
                videolabel.setText(videosource)
                videolabel.move(100 * self.count + 2, 806)
                self.file_handle.write(filename + "," + str(hour)+':'+str(minute)+':'+str(second) + "," + str((x1 + x2) / 2) + "," + str((y1 + y2) / 2) + "," + str(distance.numpy()[0] * 100))
                self.file_handle.write("\n")
                self.ui.layout.addWidget(similaritylabel, 0, self.count)
                self.ui.layout.addWidget(targetlabel, 1, self.count)
                self.ui.layout.addWidget(timelabel, 2, self.count)
                self.ui.layout.addWidget(videolabel, 3, self.count)
                self.ui.scrollAreaWidgetContents.setLayout(self.ui.layout)
                self.count += 1
            else:
                continue
        frame = self.drawObjectInfo(frame, person_boxs, person_class_names)
        return frame

    def getTime(self, currframenum,frameRate):
        totalseconds = int(currframenum / frameRate)
        second = totalseconds % 60
        totalminutes = int(totalseconds / 60)
        minute = totalminutes % 60
        hour = int(totalminutes / 60)
        return hour, minute, second

    def preprocess(self,detections,frameheight,framewidth, imghw):
        # 图像坐标 x, y, w, h转换
        person_boxs = []
        person_class_names = []
        for detection in detections:
            x, y, w, h = detection[0], \
                         detection[1], \
                         detection[2] - detection[0], \
                         detection[3] - detection[1]
            x = x * framewidth / imghw[1]
            y = y * frameheight / imghw[0]
            w = w * framewidth / imghw[1]
            h = h * frameheight / imghw[0]
            # x = x - w / 2
            # y = y - h / 2
            if int(detection[-1]) == 0:
                person_boxs.append([x, y, w, h])
                person_class_names.append("person " + str(round(detection[-2] * 100, 2)))
        return person_boxs, person_class_names

    def drawObjectInfo(self,image, objboxs, objclassnames):
        # 画图，标记图像信息
        # n=0
        for box, class_name in zip(objboxs, objclassnames):
            addtext = class_name
            # 将 getsize 替换为 getbbox
            bbox = self.font.getbbox(addtext)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x, y, w, h = box
            x = int(round(x))
            y = int(round(y))
            w = int(round(w))
            h = int(round(h))
            image = np.array(image)
            image = np.int8(image)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(image, (x, y - text_h), (x + text_w, y), (255, 0, 0), -1)
            image = np.uint8(image)
            img = Image.fromarray(image)
            draw = ImageDraw.Draw(img)
            draw.text((x, y - text_h), addtext, (255, 255, 255), font=self.font)
            image = np.asarray(img)
        return image

    def Extract(self, image):
        # 提取MGN特征
        self.ensure_reid_model_loaded()
        self.model_ReID.eval()
        with torch.no_grad():
            image = self.img_to_tensor(image, self.img_transform)
            image = image.to(self.device)
            outputs = self.model_ReID(image)
            feature = outputs[0].data.cpu()
            # 确保特征格式一致性
            if isinstance(feature, torch.Tensor) and feature.dim() > 1:
                # 如果是多维张量，展平为一维
                feature = feature.reshape(1, -1)
            # print(feature.shape)
        return feature
# 计算特征的余弦相似度进行匹配
    def Align(self, q_feature, t_feature):
        try:
            # 确保两个特征向量维度匹配
            if q_feature.shape != t_feature.shape and q_feature.dim() == t_feature.dim():
                # 如果维度不匹配但维度数相同，尝试调整大小
                if q_feature.dim() == 2 and t_feature.dim() == 2:
                    # 如果都是2D，调整为相同的大小
                    t_feature = F.interpolate(t_feature.unsqueeze(0), size=q_feature.shape[1], mode='linear').squeeze(0)
            
            # 归一化特征向量
            q_feature = F.normalize(q_feature)
            t_feature = F.normalize(t_feature)
            
            # 计算余弦相似度
            distance = F.cosine_similarity(q_feature, t_feature, dim=1)
            return distance
        except Exception as e:
            print(f"计算特征相似度时出错: {e}")
            # 返回一个低相似度的值
            return torch.tensor([0.0])

    def img_to_tensor(self, img, transform):
        img = transform(img)
        img = img.unsqueeze(0)
        return img

    def ensure_reid_model_loaded(self):
        """确保ReID模型已加载"""
        if self.model_ReID is None:
            # 加载REID模型
            print("加载ReID模型...")
            ckpt = utility.checkpoint(args)
            ckpt.dir = "./MGN/output/mgn/"
            self.model_ReID = reidModel(args, ckpt)
            print("ReID模型加载完成")
        
    def ensure_yolo_model_loaded(self):
        """确保YOLO模型已加载"""
        if self.model is None:
            print("加载YOLO模型...")
            # 加载YOLO
            self.yolo_weights = "./yolov5/weights/yolov5x.pt"
            imgsz = 640
            self.conf_thres = 0.3
            self.iou_thres = 0.5
            self.classes = 0
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
            self.model = attempt_load(self.yolo_weights, map_location=self.device)  # load FP32 model
            self.stride = int(self.model.stride.max())  # model stride
            self.img_size = check_img_size(imgsz, s=self.stride)  # check img_size
            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
            if self.half:
                self.model.half()  # to FP16
            print("YOLO模型加载完成")

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)

    def update_status(self, message):
        """更新界面状态信息"""
        # 更新状态标签
        if hasattr(self.ui, 'status_label'):
            self.ui.status_label.setText(message)
            
        # 更新状态栏
        if hasattr(self.mainWnd, 'statusBar'):
            try:
                self.mainWnd.statusBar().showMessage(message)
            except Exception as e:
                print(f"更新状态栏失败: {e}")
                pass

    def openCamera(self):
        """打开摄像头并进行实时检测"""
        # 检查是否已选择了目标人员
        if not hasattr(self, 'q_feature'):
            QMessageBox.warning(self.mainWnd, "提示", "请先选择目标人员图片")
            return
            
        try:
            # 尝试打开摄像头
            self.cap = cv2.VideoCapture(0)  # 0表示默认摄像头
            
            if not self.cap.isOpened():
                QMessageBox.warning(self.mainWnd, "错误", "无法打开摄像头，请检查设备连接")
                return
                
            # 设置摄像头分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
            # 获取帧率
            self.frameRate = self.cap.get(cv2.CAP_PROP_FPS)
            if self.frameRate <= 0 or self.frameRate > 60:
                self.frameRate = 30  # 设置默认帧率
                
            # 设置处理帧率（每隔几帧处理一次）
            self.process_frameRate = 5
                
            # 更新UI状态
            self.ui.select_video.setEnabled(False)
            self.ui.url_video.setEnabled(False)
            self.ui.camera_button.setEnabled(False)
            self.ui.stop_detect.setEnabled(True)
            self.update_status("摄像头已启动，正在进行实时检测...")
                
            # 启动处理线程
            th = threading.Thread(target=self.process_camera)
            th.start()
                
        except Exception as e:
            QMessageBox.warning(self.mainWnd, "错误", f"启动摄像头失败: {str(e)}")
            
    def process_camera(self):
        """处理摄像头视频流"""
        # 打开文件记录检测结果
        self.file_handle = open('camera_detection.txt', mode='w')
        currframenum = 0
        
        # 设置状态显示标签
        self.ui.status_label.setText("摄像头已启动，等待检测...")
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                currframenum += 1
                
                # 镜像翻转，使摄像头显示效果更直观
                frame = cv2.flip(frame, 1)
                
                # 只处理帧间隔为process_interval的帧，提高性能
                if currframenum % self.process_frameRate == 0:
                    # 进行人员检测和识别
                    processed_frame = self.camera_detect(frame, currframenum)
                    processed_frame = cv2.resize(processed_frame, (1000, 563))
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                    h, w, c = processed_frame.shape
                    img = QImage(processed_frame, w, h, 3 * w, QImage.Format_RGB888)
                    self.ui.target_video.setPixmap(QPixmap.fromImage(img))
                else:
                    # 不处理的帧直接显示
                    frame = cv2.resize(frame, (1000, 563))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, c = frame.shape
                    img = QImage(frame, w, h, 3 * w, QImage.Format_RGB888)
                    self.ui.target_video.setPixmap(QPixmap.fromImage(img))
                
                # 检查是否需要停止
                if self.stopEvent.is_set():
                    self.stopEvent.clear()
                    break
                    
                # 控制帧率，避免CPU占用过高
                cv2.waitKey(10)
            else:
                # 如果读取失败，尝试重新连接
                time.sleep(0.5)
                continue
        
        # 清理资源
        self.cap.release()
        self.file_handle.close()
        self.ui.target_video.clear()
        self.ui.stop_detect.setEnabled(False)
        self.ui.select_video.setEnabled(True)
        self.ui.url_video.setEnabled(True)
        self.ui.camera_button.setEnabled(True)
        self.update_status("摄像头已关闭")
    
    def camera_detect(self, frame, frame_num):
        """处理摄像头中的单帧图像"""
        # 确保YOLO模型已加载
        self.ensure_yolo_model_loaded()
        
        # 使用和YOLO函数相同的图像预处理
        img = letterbox(frame, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 执行推理
        pred = self.model(img, augment=False)[0]
        # 应用NMS
        imghw = [img.shape[2], img.shape[3]]
        detections = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=False)
        detections = detections[0].cpu().numpy()
        
        # 预处理检测结果
        person_boxs, person_class_names = self.preprocess(detections, frame.shape[0], frame.shape[1], imghw)
        
        # 更新状态信息
        self.ui.status_label.setText(f"检测中... 已检测到 {len(person_boxs)} 人")
        
        # 使用PIL绘制中文文本
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        
        # 添加时间戳
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        draw.text((10, 30), timestamp, fill=(0, 255, 0), font=self.font)
        
        # 添加检测状态信息
        status_text = f"检测中... 帧: {frame_num}, 人数: {len(person_boxs)}"
        draw.text((10, 60), status_text, fill=(0, 255, 0), font=self.font)
        
        # 将PIL图像转回OpenCV格式
        frame = cv2.cvtColor(np.asarray(frame_pil), cv2.COLOR_RGB2BGR)
        
        # 遍历检测到的每个行人
        matched_count = 0
        for box in person_boxs:
            x1 = int(box[0])
            x2 = int(box[0] + box[2])
            y1 = int(box[1])
            y2 = int(box[1] + box[3])
            
            # 确保坐标在图像范围内
            x1 = max(0, x1)
            x2 = min(frame.shape[1], x2)
            y1 = max(0, y1)
            y2 = min(frame.shape[0], y2)
            
            # 提取行人区域
            person = frame[y1:y2, x1:x2]
            if person.size == 0:  # 检查裁剪区域是否为空
                continue
                
            # 转换为PIL图像并提取特征
            person_img = Image.fromarray(cv2.cvtColor(person, cv2.COLOR_BGR2RGB))
            t_feature = self.Extract(person_img)
            
            # 计算与目标人物的相似度
            similarity = self.Align(self.q_feature, t_feature)
            similarity_value = similarity.item() if hasattr(similarity, 'item') else float(similarity)
            
            # 与数据库中的所有ID进行匹配，找到最佳匹配
            best_match_id = None
            best_match_similarity = 0
            
            # 首先检查与当前选择的目标人物的匹配情况
            if similarity_value > self.thres:
                best_match_id = "目标人员"
                best_match_similarity = similarity_value
            
            # 然后与数据库中的所有人员比较
            for person_id, feature_data in self.feature_db.items():
                feature = feature_data['feature']
                db_similarity = self.Align(feature, t_feature)
                db_similarity_value = db_similarity.item() if hasattr(db_similarity, 'item') else float(db_similarity)
                
                # 如果数据库中有更好的匹配，则更新最佳匹配ID
                if db_similarity_value > best_match_similarity and db_similarity_value > self.thres:
                    best_match_id = person_id
                    best_match_similarity = db_similarity_value
            
            # 使用PIL绘制文本
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            
            # 根据是否匹配成功绘制不同颜色的框和标签
            if best_match_id:
                # 在图像上标记匹配到的行人 - 绿色框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # 在人员框上方绘制背景色块
                label_width = max(len(best_match_id) * 10 + 60, 100)  # 根据ID长度动态调整宽度
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + label_width, y1), (0, 255, 0), -1)  # 绿色背景
                
                # 显示ID和相似度
                id_text = f"ID:{best_match_id} {best_match_similarity*100:.1f}%"
                draw.text((x1 + 5, y1 - 28), id_text, fill=(0, 0, 0), font=self.font)  # 黑色文字
                
                # 记录日志
                self.file_handle.write(f"{timestamp},{x1},{y1},{x2},{y2},{best_match_id},{best_match_similarity*100:.1f}%\n")
                
                # 更新状态栏信息
                match_info = f"找到匹配! ID: {best_match_id}, 相似度: {best_match_similarity*100:.1f}%"
                self.update_match_info(match_info)
                
                # 将匹配的人员添加到结果列表
                if matched_count < 5:  # 限制最多显示5个匹配结果，避免界面过于拥挤
                    self.add_match_to_results(person, best_match_similarity, timestamp, best_match_id)
                    matched_count += 1
                    print(f"添加匹配结果: ID={best_match_id}, 相似度={best_match_similarity*100:.1f}%")
            else:
                # 对于未匹配的行人，使用蓝色边框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # 显示相似度
                if similarity_value > 0.6:  # 只显示相似度超过一定阈值的
                    low_match_text = f"未知 {similarity_value*100:.1f}%"
                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + 100, y1), (255, 0, 0), -1)  # 蓝色背景
                    draw.text((x1 + 5, y1 - 28), low_match_text, fill=(255, 255, 255), font=self.font)  # 白色文字
            
            # 将PIL图像转回OpenCV格式
            frame = cv2.cvtColor(np.asarray(frame_pil), cv2.COLOR_RGB2BGR)
        
        # 如果没有匹配到任何人，更新状态信息
        if matched_count == 0 and len(person_boxs) > 0:
            self.update_match_info("未找到匹配的目标人员")
        
        return frame
        
    def update_match_info(self, info):
        """更新匹配信息到状态栏和结果标签"""
        # 更新状态栏
        if hasattr(self.mainWnd, 'statusBar'):
            try:
                self.mainWnd.statusBar().showMessage(info)
            except Exception as e:
                print(f"更新状态栏失败: {e}")
                
        # 更新结果标签
        result_info = f"检测结果: {info}"
        self.ui.result_label.setText(result_info)
        
        # 更新匹配信息标签
        if hasattr(self.ui, 'match_info_label'):
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            match_detail = f"[{timestamp}] {info}"
            self.ui.match_info_label.setText(match_detail)
        
        # 设置结果标签样式以突出显示
        if "找到匹配" in info:
            self.ui.result_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #0f9d58; background-color: #f5f5f5; border-bottom: 1px solid #dddddd;")
            if hasattr(self.ui, 'match_info_label'):
                self.ui.match_info_label.setStyleSheet("font-size: 13px; color: #0f9d58; font-weight: bold; background-color: transparent; border: none;")
        else:
            self.ui.result_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4a86e8; background-color: #f5f5f5; border-bottom: 1px solid #dddddd;")
            if hasattr(self.ui, 'match_info_label'):
                self.ui.match_info_label.setStyleSheet("font-size: 13px; color: #333333; font-weight: bold; background-color: transparent; border: none;")

    def add_match_to_results(self, person_img, similarity, timestamp, person_id="目标人员"):
        """将匹配的人员添加到结果列表"""
        # 确保清除之前的结果（如果需要）
        if self.count >= 10:  # 如果结果过多，清除之前的结果
            self.Clear()
            
        try:
            # 创建相似度标签
            similaritylabel = QLabel()
            similaritylabel.setText(f"ID: {person_id}\n相似度: {similarity*100:.2f}%")
            similaritylabel.setStyleSheet("color: #0f9d58; font-weight: bold;")
            
            # 创建人员图像标签
            targetlabel = QLabel()
            targetlabel.setFixedSize(128, 256)
            
            # 调整图像大小
            if isinstance(person_img, np.ndarray):
                # 如果是ndarray格式，先转成RGB
                person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            else:
                # 如果是PIL图像，转为ndarray
                person_img = np.array(person_img)
                
            person_img = cv2.resize(person_img, (128, 256))
            h, w, c = person_img.shape
            
            # 转换为QImage并显示
            targetperson = QImage(person_img, w, h, 3 * w, QImage.Format_RGB888)
            targetlabel.setPixmap(QPixmap.fromImage(targetperson))
            
            # 创建时间标签
            timelabel = QLabel()
            timelabel.setText(f'检测时间：{timestamp}')
            
            # 创建来源标签
            sourcelabel = QLabel()
            sourcelabel.setText(f'来源: 摄像头实时检测 | ID: {person_id}')
            
            # 将所有标签添加到布局
            self.ui.layout.addWidget(similaritylabel, 0, self.count)
            self.ui.layout.addWidget(targetlabel, 1, self.count)
            self.ui.layout.addWidget(timelabel, 2, self.count)
            self.ui.layout.addWidget(sourcelabel, 3, self.count)
            self.ui.scrollAreaWidgetContents.setLayout(self.ui.layout)
            
            # 显示匹配信息
            match_info = f"匹配结果 #{self.count+1} - ID: {person_id}, 相似度: {similarity*100:.2f}%"
            self.ui.result_label.setText(match_info)
            
            # 在匹配信息面板中显示详细信息
            if hasattr(self.ui, 'match_info_label'):
                detail_info = f"[{timestamp}] 成功匹配: ID={person_id}, 相似度: {similarity*100:.2f}%"
                self.ui.match_info_label.setText(detail_info)
                self.ui.match_info_label.setStyleSheet("font-size: 13px; color: #0f9d58; font-weight: bold; background-color: transparent; border: none;")
            
            self.count += 1
            print(f"成功添加匹配结果到位置: {self.count-1}")
            
            # 确保结果区域滚动到最新位置
            self.ui.scrollArea.ensureVisible(0, self.count * 300)
            
        except Exception as e:
            print(f"添加匹配结果时出错: {e}")
            self.update_status(f"添加匹配结果时出错: {e}")

def add_bilibili_support():
    """添加对Bilibili视频的支持"""
    try:
        from you_get import common as you_get
        
        # 可以添加一个检测是否为Bilibili URL的函数
        def is_bilibili_url(url):
            return "bilibili.com" in url
            
        # 获取真实视频URL的函数
        def get_real_url(bilibili_url):
            # 这里需要实现提取直链的代码
            # you_get库可以实现此功能
            # 返回直链
            pass
    except ImportError:
        print("需要安装you-get库来支持Bilibili视频")

def get_stream_url(url):
    """
    尝试获取视频平台的直接流链接
    目前支持：常见RTMP/RTSP/HTTP流以及尝试解析一些常见的视频网站
    返回: 直接可播放的URL或原始URL
    """
    import re
    
    # 如果本身就是直接流，直接返回
    if url.startswith(('rtmp://', 'rtsp://', 'http://', 'https://')) and any(ext in url for ext in ['.m3u8', '.mp4', '.flv', '.ts']):
        return url
        
    # 常见公开测试流
    test_streams = {
        'test1': 'http://vjs.zencdn.net/v/oceans.mp4',
        'test2': 'http://clips.vorwaerts-gmbh.de/big_buck_bunny.mp4',
        'cctv1': 'http://ivi.bupt.edu.cn/hls/cctv1hd.m3u8',
        'cctv3': 'http://ivi.bupt.edu.cn/hls/cctv3hd.m3u8',
        'cctv5': 'http://ivi.bupt.edu.cn/hls/cctv5hd.m3u8',
        'demo': 'rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4'
    }
    
    # 检查是否是测试流关键字
    if url.lower() in test_streams:
        return test_streams[url.lower()]
    
    # 尝试解析视频网站
    try:
        # Bilibili
        if 'bilibili.com' in url or 'b23.tv' in url:
            try:
                from you_get import common as you_get
                # 这里只做简单检测，实际使用需要更复杂的处理
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(None, "提示", "Bilibili视频需要下载后处理，请使用本地视频文件夹功能")
            except ImportError:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(None, "提示", "需要安装you-get库来支持Bilibili视频")
            return None
    except Exception as e:
        print(f"解析视频URL失败: {e}")
    
    # 无法处理的情况下返回原始URL
    return url