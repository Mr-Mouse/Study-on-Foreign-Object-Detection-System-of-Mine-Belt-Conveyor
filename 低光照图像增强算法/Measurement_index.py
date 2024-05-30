"""
File: Measurement_index.py
Author: CUMT-Muzihao
Date: 2024/05/06
Description:
"""
import math
import os
import random
from time import time

import cv2
import numpy as np
from LLIE import *
from skimage.metrics import structural_similarity
from tqdm import tqdm


def MSE(img, img_enhance):
    img = img.astype(np.float64)
    img_enhance = img_enhance.astype(np.float64)
    img_error = (img - img_enhance) ** 2
    mse = np.mean(img_error)
    return mse


def PSNR(img, img_enhance):
    img = img.astype(np.float64)
    img_enhance = img_enhance.astype(np.float64)
    mse = np.mean((img - img_enhance) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def SSIM(img, img_enhance):
    ssim_b = structural_similarity(img[:, :, 0], img_enhance[:, :, 0], full=False)
    ssim_g = structural_similarity(img[:, :, 1], img_enhance[:, :, 1], full=False)
    ssim_r = structural_similarity(img[:, :, 2], img_enhance[:, :, 2], full=False)

    # 计算平均 SSIM 值
    ssim = (ssim_r + ssim_g + ssim_b) / 3.0
    return ssim


def LOE(img, img_enhance):
    img = cv2.resize(img, [100, 100])
    img_enhance = cv2.resize(img_enhance, [100, 100])
    H, W, C = img.shape
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_enhance = cv2.normalize(img_enhance, None, 0, 255, cv2.NORM_MINMAX)
    # 得到RGB通道中的最大值
    L = np.max(img, axis=2)
    L_e = np.max(img_enhance, axis=2)

    # 计算相对亮度顺序差 RD
    def RD(x, y):
        mat = np.ones((H, W)) * L[x, y]
        mat_en = np.ones((H, W)) * L_e[x, y]
        res = np.bitwise_xor(np.where(mat >= L, 1, 0), np.where(mat_en >= L_e, 1, 0))
        return np.sum(res)

    # 计算亮度顺序误差 LOE
    loe = 0
    for i in range(H):
        for j in range(W):
            loe += RD(i, j)
    loe = round(loe / (H * W), 2)
    return loe


def Consumption_time(func, imgs, img_infer):

    if func.__name__ == 'HS_transform':
        t = time()
        for i in range(100):
            func(imgs[i], img_infer)
        t = time() - t
    else:
        t = time()
        for i in range(100):
            func(imgs[i])
        t = time() - t
    return t


if __name__ == '__main__':
    hightlight_path = r'./LOLdataset/high/'
    lowlight_path = r'./LOLdataset/low/'
    hightlight_images = os.listdir(hightlight_path)
    lowlight_images = os.listdir(lowlight_path)
    functions = [HE_transform, HS_transform,
                 HSV_transform, Gamma_transform, Wavelet_transform,
                 SSR_transform, MSR_transform, ]
    images_path = r'D:/active file/毕业设计-母子浩/模型代码/传送带异物检测/Datasets_Creation/Foreign_bodies_datasets/images/train/'
    image_infer = cv2.imread(r'./Reference_picture.jpg')
    imgs_name = os.listdir(images_path)
    images = [cv2.imdecode(np.fromfile(images_path+img_name, dtype=np.uint8), -1) for img_name in imgs_name]
    for function in functions:
        ctime = Consumption_time(function, images, image_infer)
        print(function.__name__.ljust(17, " ") + f":  {ctime*10:.2f}ms")
    # MSE_index = np.zeros(7)
    # PSNR_index = np.zeros(7)
    # SSIM_index = np.zeros(7)
    # LOE_index = np.zeros(7)
    #
    # image_number = 50
    # for i in tqdm(range(image_number)):
    #     image_out = []
    #     n = random.randint(0, 400)
    #     hightlight_image = cv2.imread(hightlight_path + hightlight_images[1])
    #     lowlight_image = cv2.imread(lowlight_path + lowlight_images[1])
    #
    #     for func in functions:
    #         if func.__name__ == 'HS_transform':
    #             image_out.append(func(lowlight_image, hightlight_image))
    #         else:
    #             image_out.append(func(lowlight_image))
    #     for j in range(7):
    #         MSE_index[j] += MSE(lowlight_image, image_out[j]) / image_number
    #         PSNR_index[j] += PSNR(hightlight_image, image_out[j]) / image_number
    #         SSIM_index[j] += SSIM(hightlight_image, image_out[j]) / image_number
    #         LOE_index[j] += LOE(lowlight_image, image_out[j]) / image_number
    #
    # for i in range(7):
    #     print((functions[i].__name__ + ":").ljust(20, " "), end="")
    #     print(f"MSE_index:{int(MSE_index[i])}".ljust(20, " "),
    #           f"\tSSIM_index:{SSIM_index[i]:.2f}",
    #           f"\tPSNR_index:{PSNR_index[i]:.2f}",
    #           f"\tLOE_index:{int(LOE_index[i])}")
