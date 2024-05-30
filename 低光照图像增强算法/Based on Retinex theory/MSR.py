"""
File: MSR.py
Author: CUMT-Muzihao
Date: 2024/04/14
Description:多尺度Retinex算法
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../low_light_image4.jpg')
img_S = img
img_out = []
for i in range(3):
    ln_S = np.log((img_S + 0.001) / 255.0)
    img_I = cv2.GaussianBlur(img_S, (5, 5), 0)
    ln_I = np.log((img_I + 0.001) / 255.0)
    ln_R = ln_S - ln_I * 0.4
    img_S = np.zeros_like(ln_R)
    cv2.normalize(np.exp(ln_R / 1.5), img_S, 0, 255, cv2.NORM_MINMAX)
    img_out.append(img_S)
image_enhanced = (img_out[0] + img_out[1] + img_out[2]) / 3
image_enhanced = cv2.convertScaleAbs(image_enhanced)
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('原图像', fontdict={'family': 'KaiTi', 'size': 10}), plt.axis('off')
plt.subplot(122), plt.imshow(np.uint8(image_enhanced), cmap='gray')
plt.title('增强后图像', fontdict={'family': 'KaiTi', 'size': 10}), plt.axis('off')
cv2.imwrite('low_light_image4_7.jpg', np.uint8(image_enhanced), [cv2.IMWRITE_PNG_COMPRESSION, 2])
plt.show()