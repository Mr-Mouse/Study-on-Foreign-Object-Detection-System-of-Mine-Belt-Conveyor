"""
File: Model_information.py
Author: CUMT-Muzihao
Date: 2024/05/10
Description:
"""

from ultralytics import YOLO

if __name__ == '__main__':
    Model_yolov8 = YOLO('./test_imformation/yolov8.yaml')
    Model_yolov8dw = YOLO('./test_imformation/yolov8_DW.yaml')
    Model_yolov8sa = YOLO('./test_imformation/yolov8_SA.yaml')
    Model_yolov8sai = YOLO('./test_imformation/yolov8_SAI.yaml')
    Model_yolov8saidw = YOLO('./test_imformation/yolov8_SAIDW.yaml')
    Model_yolov8saidwmini = YOLO('./test_imformation/yolov8_SAIDWmini.yaml')

    print(Model_yolov8.info(detailed=False))
    print(Model_yolov8dw.info(detailed=False))
    print(Model_yolov8sa.info(detailed=False))
    print(Model_yolov8sai.info(detailed=False))
    print(Model_yolov8saidw.info(detailed=False))
    print(Model_yolov8saidwmini.info(detailed=False))
