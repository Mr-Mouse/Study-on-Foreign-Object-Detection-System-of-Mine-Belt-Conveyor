"""
File: Datasets_select.py
Author: CUMT-Muzihao
Date: 2024/3/20
Description: 从基础数据集当中选取一部分数据进行保存.
"""
import random
import cv2
import os

image_source_path = r'./CUMT-BelT/test/'
image_save_path = r'./Foreign_bodies_datasets/images/val/'
image_names = os.listdir(image_source_path)
total_source = len(image_names)

for i in image_names:
    oldpath = os.path.join(image_source_path, i)
    new_string = i
    if '大块测试集' in i:
        new_string = i.replace('大块测试集', 'Gangue')
    elif '正常煤流测试集' in i:
        new_string = i.replace('正常煤流测试集', 'Normal')
    elif '锚杆测试集' in i:
        new_string = i.replace('锚杆测试集', 'Anchor_rod')
    newpath = os.path.join(image_source_path, new_string)
    os.rename(oldpath, newpath)
selected_index = []
image_names = os.listdir(image_source_path)
# 随机选择500帧
for _ in range(100):
    # 生成随机帧索引
    random_source_index = random.randint(0, total_source - 1)

    # 确保选取的帧索引不重复
    while random_source_index in selected_index:
        random_source_index = random.randint(0, total_source - 1)

    # 将选取的帧索引添加到列表中
    selected_index.append(random_source_index)

for idx in selected_index:
    image = cv2.imread(image_source_path + image_names[idx])
    image = cv2.resize(image, [320, 320], interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(image_save_path + image_names[idx], image)
