import shutil
import torch
from torch.nn import functional as F
import cv2
import os
import numpy as np
from MGN import model
from torchvision.transforms import transforms
from PIL import ImageFont, Image, ImageDraw
from MGN.option import args
import MGN.utils.utility as utility
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams, letterbox
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_synchronized


class Search:
    def __init__(self, queryimg_path, targetvideo_path):
        self.font = ImageFont.truetype("font/platech.ttf", 20, 0)
        self.netMain = None
        self.metaMain = None
        self.altNames = None
        self.count = 0
        self.queryimg_path = queryimg_path
        self.targetvideo_path = targetvideo_path
        self.img_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.device = torch.device('cpu' if args.cpu else 'cuda')

        '''
        加载REID模型
        '''
        ckpt = utility.checkpoint(args)
        self.model_ReID = model.Model(args, ckpt)

        '''
        加载YOLO
        '''
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

    def cleardir(self, filepath):
        del_list = os.listdir(filepath)
        for f in del_list:
            file_path = os.path.join(filepath, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    def search(self):
        self.cleardir('./test_data/target_persons/')
        self.cleardir('./test_data/targets/')
        if self.queryimg_path != '':
            person = cv2.imread(self.queryimg_path)
            person_img = Image.fromarray(person)
            self.q_feature = self.Extract(person_img)
        else:
            print("待查寻行人路径为空,请检查！")
            return
        if self.targetvideo_path != '':
            self.cap = cv2.VideoCapture(self.targetvideo_path)
            self.frameRate = self.cap.get(cv2.CAP_PROP_FPS)
            currframenum = 1
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if success:
                    if currframenum % (self.frameRate / 4) != 0:
                        currframenum += 1
                        continue
                    frame = self.YOLO(frame, currframenum)
                else:
                    break
                currframenum += 1
        else:
            print("待搜索视频路径为空,请检查！")
            return

    def YOLO(self, frame, currframenum):
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
        person_boxs, person_class_names = self.preprocess(detections, frame.shape[0], frame.shape[1], imghw)
        plot_flag = 0
        for box, class_name in zip(person_boxs, person_class_names):
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
            person = frame[y1:y2, x1:x2]
            person_img = Image.fromarray(person)
            if self.queryimg_path != '':
                t_feature = self.Extract(person_img)
                distance = self.Align(self.q_feature, t_feature)
                if distance > 0.75:
                    plot_flag = 1
                    self.count += 1
                    print('------person' + str(self.count) + '------')
                    print("相似度: %.2f%%" % ((distance) * 100))
                    hour, minute, second = self.getTime(currframenum)
                    print("出现时间: " + str(hour) + ':' + str(minute) + ':' + str(second))
                    cv2.imwrite('./test_data/target_persons/person' + str(self.count) + '.jpg', person)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    addtext = class_name
                    text_w, text_h = self.font.getsize(addtext)
                    cv2.putText(frame, addtext, (x1, y1 - text_h), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        if plot_flag == 1:
            cv2.imwrite('./test_data/targets/target' + str(currframenum) + '.jpg', frame)
        return frame

    def getTime(self, currframenum):
        totalseconds = int(currframenum / self.frameRate)
        second = totalseconds % 60
        totalminutes = int(totalseconds / 60)
        minute = totalminutes % 60
        hour = int(totalminutes / 60)
        return hour, minute, second

    def preprocess(self, detections, frameheight, framewidth, imghw):
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

    def drawObjectInfo(self, image, objboxs, objclassnames):
        # n=0
        for box, class_name in zip(objboxs, objclassnames):
            addtext = class_name
            text_w, text_h = self.font.getsize(addtext)
            x, y, w, h = box
            x = int(round(x))
            y = int(round(y))
            w = int(round(w))
            h = int(round(h))
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(image, (x, y - text_h), (x + text_w, y), (255, 0, 0), -1)
            img = Image.fromarray(image)
            draw = ImageDraw.Draw(img)
            draw.text((x, y - text_h), addtext, (255, 255, 255), font=self.font)
            image = np.asarray(img)
        return image

    def Extract(self, image):
        self.model_ReID.eval()
        with torch.no_grad():
            image = self.img_to_tensor(image, self.img_transform)
            image = image.to(self.device)
            outputs = self.model_ReID(image)
            feature = outputs[0].data.cpu()
            # print(feature.shape)
        return feature

    def Align(self, q_feature, t_feature):
        q_feature = F.normalize(q_feature)
        t_feature = F.normalize(t_feature)
        # distance = F.pairwise_distance(q_feature, t_feature, p=2)
        distance = F.cosine_similarity(q_feature, t_feature, dim=1)
        # print(distance)
        return distance

    def img_to_tensor(self, img, transform):
        img = transform(img)
        img = img.unsqueeze(0)
        return img

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)


if __name__ == "__main__":
    process = Search('./test_data/test_img/2.jpg', './test_data/test_video/target.mp4')
    process.search()