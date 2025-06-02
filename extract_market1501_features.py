import os
import torch
import pickle
from torchvision import transforms
from PIL import Image
import numpy as np
from MGN.model import Model as reidModel
from MGN.option import args
import MGN.utils.utility as utility
import shutil

# 1. 加载模型
ckpt = utility.checkpoint(args)
ckpt.dir = "./MGN/output/mgn/"
args.model = "mgn"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = reidModel(args, ckpt)
model.eval()
model.to(device)

# 2. 定义图片预处理
img_transform = transforms.Compose([
    transforms.Resize((384, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_feature(img_path):
    """提取图像特征"""
    img = Image.open(img_path).convert('RGB')
    img = img_transform(img)
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(img)[0].cpu()
    return feat

def process_dataset(data_dir, save_dir, dataset_type='query'):
    """处理数据集并构建特征库"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
    
    # 用于存储所有特征的字典
    feature_db = {}
    
    # 遍历数据集目录
    for img_name in os.listdir(data_dir):
        if img_name.endswith('.jpg'):
            # 解析图像信息
            # 格式示例: 0002_c1s1_000451_00.jpg
            parts = img_name.split('_')
            if len(parts) >= 4:
                person_id = parts[0]  # 行人ID
                camera_id = parts[1]  # 摄像头ID
                frame_id = parts[2]   # 帧号
                
                # 构建图像路径
                img_path = os.path.join(data_dir, img_name)
                
                # 提取特征
                try:
                    feature = extract_feature(img_path)
                    
                    # 复制图像到特征库目录
                    new_img_path = os.path.join(save_dir, 'images', img_name)
                    shutil.copy2(img_path, new_img_path)
                    
                    # 构建特征数据
                    feature_data = {
                        'feature': feature,
                        'img_path': new_img_path,
                        'person_id': person_id,
                        'camera_id': camera_id,
                        'frame_id': frame_id,
                        'dataset_type': dataset_type
                    }
                    
                    # 使用person_id作为键存储特征
                    if person_id not in feature_db:
                        feature_db[person_id] = []
                    feature_db[person_id].append(feature_data)
                    
                    print(f"处理完成: {img_name}")
                    
                except Exception as e:
                    print(f"处理 {img_name} 时出错: {e}")
    
    # 保存特征库
    feature_db_path = os.path.join(save_dir, 'market1501_features.pkl')
    with open(feature_db_path, 'wb') as f:
        pickle.dump(feature_db, f)
    
    print(f"\n特征库构建完成！")
    print(f"特征库路径: {feature_db_path}")
    print(f"包含 {len(feature_db)} 个不同ID的行人特征")
    
    # 统计信息
    total_images = sum(len(features) for features in feature_db.values())
    print(f"总共处理了 {total_images} 张图像")
    
    return feature_db

if __name__ == '__main__':
    # 设置路径
    base_dir = r'd:\行人重识别\reid_system\MGN\market1501'
    save_dir = r'd:\行人重识别\person_db_features'
    
    # 处理查询集
    query_dir = os.path.join(base_dir, 'query')
    print("开始处理查询集...")
    query_db = process_dataset(query_dir, save_dir, 'query')
    
    # 处理训练集
    train_dir = os.path.join(base_dir, 'bounding_box_train')
    print("\n开始处理训练集...")
    train_db = process_dataset(train_dir, save_dir, 'train')
    
    # 合并特征库
    combined_db = {**query_db, **train_db}
    combined_db_path = os.path.join(save_dir, 'market1501_combined_features.pkl')
    with open(combined_db_path, 'wb') as f:
        pickle.dump(combined_db, f)
    
    print(f"\n合并特征库完成！")
    print(f"合并特征库路径: {combined_db_path}")
    print(f"合并后包含 {len(combined_db)} 个不同ID的行人特征")
    total_images = sum(len(features) for features in combined_db.values())
    print(f"总共包含 {total_images} 张图像的特征")