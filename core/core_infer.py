import torch
from MGN.model import Model as reidModel
from MGN.option import args
import MGN.utils.utility as utility
from torchvision.transforms import transforms
from PIL import Image
from torch.nn import functional as F
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import check_img_size, non_max_suppression
from yolov5.utils.torch_utils import time_synchronized
import numpy as np
import cv2
import math
import base64
import os

class PersonReIDInfer:
    def __init__(self, model_type="MGN"):
        self.img_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.thres = 0.87 if model_type == "MGN" else 0.825
        ckpt = utility.checkpoint(args)
        if model_type == "MGN":
            ckpt.dir = "./MGN/output/mgn/"
            args.model = "mgn"
        else:
            ckpt.dir = "./MGN/output/base/"
            args.model = "base"
        self.model_ReID = reidModel(args, ckpt)
        self.model_ReID.eval()
        # 加载YOLO
        self.yolo_weights = "./yolov5/weights/yolov5x.pt"
        imgsz = 640
        self.conf_thres = 0.3
        self.iou_thres = 0.5
        self.classes = 0
        self.half = self.device.type != 'cpu'
        self.model = attempt_load(self.yolo_weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(imgsz, s=self.stride)
        if self.half:
            self.model.half()

    def extract_feature(self, image: Image.Image):
        with torch.no_grad():
            img_tensor = self.img_transform(image).unsqueeze(0).to(self.device)
            outputs = self.model_ReID(img_tensor)
            feature = outputs[0].data.cpu()
        return feature

    def align(self, q_feature, t_feature):
        q_feature = F.normalize(q_feature)
        t_feature = F.normalize(t_feature)
        distance = F.cosine_similarity(q_feature, t_feature, dim=1)
        return distance

    def detect_person(self, frame):
        img = letterbox(frame, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            pred = self.model(img, augment=False)[0]
            imghw = [img.shape[2], img.shape[3]]
            detections = non_max_suppression(
                pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=False)
            detections = detections[0].cpu().numpy()
        return detections, imghw

    def get_time(self, currframenum, frameRate):
        totalseconds = int(currframenum / frameRate)
        second = totalseconds % 60
        totalminutes = int(totalseconds / 60)
        minute = totalminutes % 60
        hour = int(totalminutes / 60)
        return hour, minute, second