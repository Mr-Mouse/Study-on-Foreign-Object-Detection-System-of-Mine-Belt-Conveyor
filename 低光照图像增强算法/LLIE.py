"""
File: LLIE.py(low-light image enhance)
Author: CUMT-Muzihao
Date: 2024/04/14
Description:低光照图像增强算法合集
"""
from time import time

import cv2
import numpy as np
import pywt
from matplotlib import pyplot as plt

__all__ = ('HE_transform', 'HS_transform',
           'HSV_transform', 'Gamma_transform', 'Wavelet_transform',
           'SSR_transform', 'MSR_transform',)


# 直方图均衡化图片亮度增强 histogram equalization transform
def HE_transform(image_in):
    B, G, R = cv2.split(image_in)
    BGR_out = []
    for channel in [B, G, R]:
        BGR_out.append(cv2.equalizeHist(channel))
    image_out = cv2.merge(BGR_out)
    image_out = cv2.convertScaleAbs(image_out)
    return image_out


# 直方图规定化图片亮度增强 Histogram specification transform
def HS_transform(image_in, image_refer):
    # 计算原图像和参考图像的直方图
    hist_in = cv2.calcHist([image_in], [0], None, [256], [0, 256])
    his_refer = cv2.calcHist([image_refer], [0], None, [256], [0, 256])

    # 将直方图归一化
    hist_in = hist_in / float(np.sum(hist_in))
    his_refer = his_refer / float(np.sum(his_refer))
    # 计算原图像和参考图像的累积分布函数（CDF）
    cdf1 = hist_in.cumsum()
    cdf2 = his_refer.cumsum()

    # 计算灰度值映射
    lut = np.interp(cdf1, cdf2, np.arange(0, 256))

    # 针对每个像素应用灰度映射
    image_out = lut[image_in]

    return image_out


# HSV空间变换图片亮度增强 HSV transform
def HSV_transform(image_in):
    # 将图像从RGB空间转换到HSV空间
    hsv_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2HSV)
    # 分割H、S、V通道
    h, s, v = cv2.split(hsv_image)
    # 对V通道进行直方图均衡化
    equalized_v = cv2.equalizeHist(v)
    # 将增强后的V通道与原始H、S通道合并回HSV图像
    enhanced_hsv_image = cv2.merge((h, s, equalized_v))
    # 将增强后的HSV图像转换回RGB空间
    image_out = cv2.cvtColor(enhanced_hsv_image, cv2.COLOR_HSV2BGR)
    return image_out


# 伽马变换图片亮度增强 Gamma transform
def Gamma_transform(image_in):
    image_in = image_in.astype(np.float32)
    gamma = 0.3
    m = 255 ** (1 - gamma)
    image_out = cv2.pow(image_in, gamma) * m
    image_out = cv2.convertScaleAbs(image_out)
    return image_out


# 小波变换图片亮度增强 Wavelet transform
def Wavelet_transform(image_in):
    B, G, R = cv2.split(image_in)
    coeffs = [pywt.dwt2(channel, 'haar') for channel in [B, G, R]]
    alpha = 2.5
    coeffs_scaled = [(coeff[0] * alpha, coeff[1]) for coeff in coeffs]
    BGR_out = [pywt.idwt2(coeff, 'haar') for coeff in coeffs_scaled]
    image_out = cv2.merge(BGR_out)
    image_out = cv2.convertScaleAbs(image_out)
    return image_out


# 单尺度Retinex算法图像亮度增强
def SSR_transform(image_in):
    # 将图像转换为浮点型
    img_float = np.float32(image_in) / 255 + 0.01
    # 计算图像的对数域
    log_img = np.log(img_float)

    # 使用高斯滤波器平滑图像
    img_blur = cv2.GaussianBlur(log_img, (5, 5), 0)
    # 计算Retinex
    retinex = log_img - img_blur * 0.4

    # 归一化
    image_out = cv2.normalize(np.exp(retinex / 1.5), None, 0, 255, cv2.NORM_MINMAX)
    image_out = cv2.convertScaleAbs(image_out)

    return image_out


# 多尺度Retinex算法图像亮度增强
def MSR_transform(image_in):
    # 将图像转换为浮点型
    img_float = np.float32(image_in) / 255 + 0.01
    # 计算图像的对数域
    log_img = np.log(img_float)
    img_blurs = []
    for i in range(3):
        # 使用高斯滤波器平滑图像
        img_blurs.append(cv2.GaussianBlur(log_img, (5 + 10 * i, 5 + 10 * i), 0) * 0.4)
    # 计算Retinex
    retinexs = [(log_img - img_blur) for img_blur in img_blurs]
    img_out = [cv2.normalize(np.exp(retinex / 1.5), None, 0, 255, cv2.NORM_MINMAX) for retinex in retinexs]
    image_out = (img_out[0] + img_out[1] + img_out[2]) / 3
    image_out = cv2.convertScaleAbs(image_out)
    return image_out


def Light_enhance(func, image_in, image_infer=None):
    plt.figure(figsize=(20, 5), dpi=100)
    n = len(image_in)
    for img, i in zip(image_in, range(n)):
        if func.__name__ == 'HS_transform':
            img_out = func(img, image_infer)
        else:
            img_out = func(img)
        plt.subplot(1, 2 * n, i + 1)
        plt.imshow(img[:, :, ::-1])
        plt.axis('off')
        plt.subplot(1, 2 * n, i + n + 1)
        plt.imshow(img_out[:, :, ::-1])
        plt.axis('off')
    plt.savefig(func.__name__ + '.png')
    plt.show()


if __name__ == '__main__':
    # image = cv2.imread('./resuorse/CUMT2.png')
    # image1 = cv2.imread('./low_light_image1.jpg')
    # image2 = cv2.imread('./low_light_image4.jpg')
    # image_ = cv2.imread('./Reference_picture.jpg')
    # Light_enhance(HE_transform, [image1, image2], image_)
    # Light_enhance(HS_transform, [image1, image2], image_)
    # Light_enhance(HSV_transform, [image1, image2], image_)
    # Light_enhance(Gamma_transform, [image1, image2], image_)
    # Light_enhance(Wavelet_transform, [image1, image2], image_)
    # Light_enhance(SSR_transform, [image1, image2], image_)
    # Light_enhance(MSR_transform, [image1, image2], image_)

    # image_HE = HE_transform(image)
    # image_HS = HS_transform(image, image_)
    # image_HSV = HSV_transform(image)
    # image_Gamma = Gamma_transform(image)
    # image_Wavelet = Wavelet_transform(image)
    # image_SSR = SSR_transform(image)
    # image_MSR = MSR_transform(image)
    #
    # plt.figure(figsize=(20, 10), dpi=100)
    # plt.subplot(131), plt.imshow(image[:, :, ::-1], cmap='gray')
    # plt.title('原图像', fontdict={'family': 'KaiTi', 'size': 25}), plt.axis('off')
    #
    # plt.subplot(242), plt.imshow(np.uint8(image_HE[:, :, ::-1]), cmap='gray')
    # plt.title('直方图均衡化变换', fontdict={'family': 'KaiTi', 'size': 25}), plt.axis('off')
    #
    # plt.subplot(243), plt.imshow(np.uint8(image_HS[:, :, ::-1]), cmap='gray')
    # plt.title('直方图规定化变换', fontdict={'family': 'KaiTi', 'size': 25}), plt.axis('off')
    #
    # plt.subplot(244), plt.imshow(np.uint8(image_HSV[:, :, ::-1]), cmap='gray')
    # plt.title('HSV空间变换', fontdict={'family': 'KaiTi', 'size': 25}), plt.axis('off')
    #
    # plt.subplot(132), plt.imshow(np.uint8(image_Gamma[:, :, ::-1]), cmap='gray')
    # plt.title('伽马变换', fontdict={'family': 'KaiTi', 'size': 25}), plt.axis('off')
    #
    # plt.subplot(133), plt.imshow(np.uint8(image_Wavelet[:, :, ::-1]), cmap='gray')
    # plt.title('小波变换', fontdict={'family': 'KaiTi', 'size': 25}), plt.axis('off')
    #
    # plt.subplot(247), plt.imshow(np.uint8(image_SSR[:, :, ::-1]), cmap='gray')
    # plt.title('单尺度Retinex', fontdict={'family': 'KaiTi', 'size': 25}), plt.axis('off')
    #
    # plt.subplot(248), plt.imshow(np.uint8(image_MSR[:, :, ::-1]), cmap='gray')
    # plt.title('多尺度Retinex', fontdict={'family': 'KaiTi', 'size': 25}), plt.axis('off')
    # plt.savefig('色彩失真.png')
    # plt.show()

    img = cv2.imread("./resuorse/CUMT2.png")
    B, G, R = cv2.split(img)
    hist_B = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_G = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_R = cv2.calcHist([img], [2], None, [256], [0, 256])
    hist_B[hist_B > 10000] = 10000
    hist_G[hist_G > 10000] = 10000
    hist_R[hist_R > 10000] = 10000
    plt.figure(figsize=(20, 10), dpi=100)
    plt.subplot(1, 2, 1)
    plt.imshow(np.uint8(img[:, :, ::-1]), cmap='gray')
    plt.title('图例', fontdict={'family': 'KaiTi', 'size': 30}), plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.plot(hist_B, 'b'), plt.plot(hist_G, 'g'), plt.plot(hist_R, 'r')
    plt.title('图例三通道直方图', fontdict={'family': 'KaiTi', 'size': 30})
    plt.savefig('矿大.png')
    plt.show()
