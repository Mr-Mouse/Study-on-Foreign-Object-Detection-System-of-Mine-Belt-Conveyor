"""
File: transition_onnx.py
Author: CUMT-Muzihao
Date: 2024/05/22
Description:
"""
from ultralytics import YOLO

model = YOLO("yolov8_saidwmini.pt")
model.export(format="onnx")  # export the model to onnx format