"""
File: Model_test.py
Author: CUMT-Muzihao
Date: 2024/03/21
Description:传送带异物识别模型测试
"""
import os
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def cv2ImgAddText(img, txt, left, top, box_color=(0, 255, 0), textSize=20):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype(
        "msyhbd.ttc", textSize, encoding="utf-8")
    text_bbox = draw.textbbox((left, top), txt, font=fontStyle)
    box_color = box_color[::-1]
    draw.rectangle((left, top, text_bbox[2], text_bbox[3]), fill=box_color)
    draw.text((left, top), txt, (255, 255, 255), font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


model = YOLO(r'D:\active file\毕业设计-母子浩\模型代码\模型评估\models\Foreign_bodies\train4\weights\best.pt')
random_colors = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 255),
]
names = ['锚杆', '矸石', '异物', '木屑']

image_path = r'../Datasets_Creation/CUMT-BelT/test/'
image_names = os.listdir(image_path)
for image_name in image_names:
    image = cv2.imread(image_path + image_name)
    image = cv2.resize(image, (640, 640))
    results = model(image)
    for r in results:
        a = r.boxes.data
        for i in range(a.shape[0]):
            data = a.cpu().numpy()[i, :]
            image = cv2.rectangle(image, (int(data[0]), int(data[1])), (int(data[2]), int(data[3])),
                                  random_colors[int(data[5])], thickness=1)
            text = names[int(data[5])] + ":{:.2f}".format(data[4])
            image = cv2ImgAddText(image, text, int(data[0]), int(data[1]), random_colors[int(data[5])], 20)
            # print(data)
        cv2.imshow('result', image)
    cv2.waitKey(0)  # 等待按下任意键关闭窗口
    cv2.destroyAllWindows()
