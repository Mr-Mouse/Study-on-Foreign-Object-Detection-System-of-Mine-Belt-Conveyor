"""
File: Model_train.py
Author: CUMT-Muzihao
Date: 2024/03/20
Description:传送带异物识别模型训练
"""
from ultralytics import YOLO


if __name__ == '__main__':
    # Model_yolov8 = YOLO('yolov8.yaml')
    Model_yolov8sai = YOLO('yolov8_SAIDWmini.yaml')
    yaml = r"D:\active file\毕业设计-母子浩\模型代码\传送带异物检测\Model_Train\coco.yaml"
    Model_yolov8sai.train(data=yaml, epochs=3, imgsz=320, save=True, save_period=25)
