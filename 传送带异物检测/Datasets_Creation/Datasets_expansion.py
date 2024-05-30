"""
File: Datasets_expansion.py
Author: CUMT-Muzihao
Date: 2024/03/20 
Description: 对训练集进行数据增强
"""
import math
import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw


def Gaussian_noise(img):
    mean = 0
    var = 0.15
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, img.shape).astype('uint8')
    n_img = cv2.add(img, gaussian)
    return n_img


def Random_occlusion(img):
    o_img = img
    size = random.randint(1, 40)

    unit = np.ones([40, 40])
    unit[0:size, 0:size] = 0
    mask = np.tile(unit, (8, 8))
    mask = np.dstack([mask] * 3)
    o_img = o_img * np.uint8(mask)
    return o_img


def Random_line(image_array):
    img = Image.fromarray(image_array)
    width, height = img.size
    center_point = (width // 2, height // 2)
    length = min(width, height)
    angle_degrees = random.randint(0, 180)
    angle_radians = math.radians(angle_degrees)
    start_point = (center_point[0] - length * math.cos(angle_radians),
                   center_point[1] - length * math.sin(angle_radians))
    end_point = (center_point[0] + length * math.cos(angle_radians),
                 center_point[1] + length * math.sin(angle_radians))
    draw = ImageDraw.Draw(img)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    w = random.randint(0, 30)
    draw.line([start_point, end_point], fill=color, width=w)
    modified_image_array = np.array(img)
    return modified_image_array


images_path = r'./Foreign_bodies_datasets/images/val/'
labels_path = r'./Foreign_bodies_datasets/labels/val/'
image_names = os.listdir(images_path)
label_names = os.listdir(labels_path)

for image_name, label_name in zip(image_names, label_names):
    image = cv2.imread(images_path + image_name)
    image = cv2.resize(image, (320, 320))  # 640*360
    with open(labels_path + label_name, "r") as file:
        label = file.read()
    image1 = Gaussian_noise(image)
    image2 = Random_line(image)
    cv2.imwrite(images_path + image_name.replace('.jpg', '') + '_1.jpg', image1, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    with open(labels_path + label_name.replace('.txt', '') + '_1.txt', "w") as file:
        file.write(label)
    cv2.imwrite(images_path + image_name.replace('.jpg', '') + '_2.jpg', image2, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    with open(labels_path + label_name.replace('.txt', '') + '_2.txt', "w") as file:
        file.write(label)
