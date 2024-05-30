"""
File: Speed_assess.py
Author: CUMT-Muzihao
Date: 2024/05/11
Description:
"""
from time import time
import numpy as np
import torch
from ultralytics import YOLO


def Calculate_FPS(model):
    all_time = []
    for _ in range(100):
        t = time()
        x = np.random.randint(0, 256, (320, 320, 3), dtype=np.uint8)
        model(x)
        # all_time.append(resualt[0].speed['preprocess']+resualt[0].speed['inference']+resualt[0].speed['postprocess'])
        all_time.append(time()-t)
    all_time[0] = all_time[1]
    return 1.0 / np.mean(all_time)


Model_yolov8 = YOLO('./models/Foreign_bodies/train_yolov8_fbody_150/weights/best.pt')
Model_yolov8dw = YOLO('./models/Foreign_bodies/train_yolov8dw_fbody_150/weights/best.pt')
Model_yolov8sai = YOLO('./models/Foreign_bodies/train_yolov8sai_fbody_150/weights/best.pt')
Model_yolov8saidw = YOLO('./models/Foreign_bodies/train_yolov8saidw_fbody_150/weights/best.pt')
# Model_yolov8saidwmini = YOLO('./models/Foreign_bodies/train_yolov8saidwmini_fbody_150/weights/best.pt')
Model_yolov8saidwmini = YOLO('./models/Foreign_bodies/best.pt')
# Model_yolov8.export(format="onnx")
# Model_yolov8dw.export(format="onnx")
# Model_yolov8sai.export(format="onnx")
# Model_yolov8saidw.export(format="onnx")
# Model_yolov8saidwmini.export(format="onnx")


Model_yolov8_fps = Calculate_FPS(Model_yolov8)
Model_yolov8dw_fps = Calculate_FPS(Model_yolov8dw)
Model_yolov8sai_fps = Calculate_FPS(Model_yolov8sai)
Model_yolov8saidw_fps = Calculate_FPS(Model_yolov8saidw)
Model_yolov8saidwmini_fps = Calculate_FPS(Model_yolov8saidwmini)
#
print(f"YOLOv8 FPS：".ljust(20, " "), f"{Model_yolov8_fps:.2f}")
print(f"YOLOv8_DW FPS：".ljust(20, " "), f"{Model_yolov8dw_fps:.2f}")
print(f"YOLOv8_SAI FPS：".ljust(20, " "), f"{Model_yolov8sai_fps:.2f}")
print(f"YOLOv8_SAIDW FPS：".ljust(20, " "), f"{Model_yolov8saidw_fps:.2f}")
print(f"YOLOv8_SAIDWmini FPS：".ljust(20, " "), f"{Model_yolov8saidwmini_fps:.2f}")

# print(Model_yolov8.info(detailed=False))
# print(Model_yolov8dw.info(detailed=False))
# print(Model_yolov8sai.info(detailed=False))
# print(Model_yolov8saidw.info(detailed=False))
# print(Model_yolov8saidwmini.info(detailed=False))