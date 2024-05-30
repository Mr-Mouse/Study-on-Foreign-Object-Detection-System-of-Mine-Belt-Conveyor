"""
File: model_val.py
Author: CUMT-Muzihao
Date: 2024/05/01
Description:
"""
from ultralytics import YOLO
model = YOLO("./models/epoch100.pt")
# 如果不设置数据，它将使用model.pt中的数据集相关yaml文件。

if __name__ == '__main__':
    metrics = model.val(data='coco.yaml')
