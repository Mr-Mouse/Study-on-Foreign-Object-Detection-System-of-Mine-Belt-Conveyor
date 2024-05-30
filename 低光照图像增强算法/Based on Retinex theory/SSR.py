"""
File: SSR.py
Author: CUMT-Muzihao
Date: 2024/04/13
Description:单尺度Retinex算法
"""
# S = I * R
# S:输入图片
# I:光照量;需要对图像进行高斯模糊得到
# R:增强后图片
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../low_light_image4.jpg')
img_S = img
ln_S = np.log((img_S + 0.001) / 255.0)
img_I = cv2.GaussianBlur(img_S, (5, 5), 0)
ln_I = np.log((img_I + 0.001) / 255.0)
ln_R = ln_S - ln_I * 0.4
image_enhanced = np.zeros_like(ln_R)
cv2.normalize(np.exp(ln_R / 1.5), image_enhanced, 0, 255, cv2.NORM_MINMAX)
image_enhanced = cv2.convertScaleAbs(image_enhanced)

plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('原图像', fontdict={'family': 'KaiTi', 'size': 10}), plt.axis('off')
plt.subplot(122), plt.imshow(np.uint8(image_enhanced), cmap='gray')
plt.title('增强后图像', fontdict={'family': 'KaiTi', 'size': 10}), plt.axis('off')
cv2.imwrite('low_light_image4_6.jpg', np.uint8(image_enhanced), [cv2.IMWRITE_PNG_COMPRESSION, 2])
plt.show()
