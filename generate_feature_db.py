import os
import glob
import torch
import pickle
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import transforms
from MGN.model import Model as reidModel
from MGN.option import args
import MGN.utils.utility as utility

def main():
    # 检查并创建特征数据库目录
    os.makedirs("person_db_features", exist_ok=True)
    
    # 加载REID模型
    device = torch.device('cpu' if args.cpu else 'cuda')
    ckpt = utility.checkpoint(args)
    ckpt.dir = "./MGN/output/mgn/"
    args.model = "mgn"
    model_ReID = reidModel(args, ckpt)
    model_ReID.eval()
    
    # 定义图像转换
    img_transform = transforms.Compose([
        transforms.Resize((384, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 读取人物图像目录
    person_dir = input("请输入人物图像目录路径: ")
    if not os.path.exists(person_dir):
        print(f"目录 {person_dir} 不存在!")
        return
    
    # 获取所有图像文件
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(glob.glob(os.path.join(person_dir, ext)))
    
    if not image_files:
        print(f"在 {person_dir} 目录中没有找到图像文件!")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理每个图像
    for img_path in image_files:
        # 获取文件名作为ID
        person_id = os.path.splitext(os.path.basename(img_path))[0]
        
        try:
            # 读取图像
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            if img is None:
                print(f"无法读取图像 {img_path}，跳过")
                continue
                
            # 转换为RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            person_img = Image.fromarray(img).convert("RGB")
            
            # 提取特征
            with torch.no_grad():
                img_tensor = img_transform(person_img).unsqueeze(0).to(device)
                outputs = model_ReID(img_tensor)
                feature = outputs[0].data.cpu()
            
            # 保存特征到文件
            feature_data = {
                'feature': feature,
                'img_path': img_path
            }
            
            # 保存到pickle文件
            with open(f"person_db_features/{person_id}.pkl", 'wb') as f:
                pickle.dump(feature_data, f)
                
            print(f"成功提取并保存 {person_id} 的特征")
            
        except Exception as e:
            print(f"处理 {img_path} 时出错: {e}")
    
    print(f"完成! 特征数据库已保存到 person_db_features 目录")

if __name__ == "__main__":
    main() 