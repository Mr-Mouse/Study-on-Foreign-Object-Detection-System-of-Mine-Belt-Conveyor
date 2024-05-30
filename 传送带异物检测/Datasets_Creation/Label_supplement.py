"""
File: Label_supplement.py
Author: CUMT-Muzihao
Date: 2024/03/20 
Description: 补充训练集中缺失的标签文件
"""
import os

images_path = r'./Foreign_bodies_datasets/images/val/'
labels_path = r'./Foreign_bodies_datasets/labels/val/'
image_names = os.listdir(images_path)
label_names = os.listdir(labels_path)
n = 0
for image_name in image_names:
    l_name = image_name.replace('.jpg', '.txt')
    if l_name not in label_names:
        with open(labels_path + l_name, "w") as file:
            file.write('')


