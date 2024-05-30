"""
File: Gamma_transform.py
Author: CUMT-Muzihao
Date: 2024/04/12
Description:伽马变换实现亮度增强
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('../low_light_image4.jpg')
img = img.astype(np.float32) / 255.0


def gamma_correction(image, gamma_):
    return np.power(image, gamma_)


gamma = 0.3
img_gamma = gamma_correction(img, gamma)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('原图像', fontdict={'family': 'KaiTi', 'size': 10})

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_gamma, cv2.COLOR_BGR2RGB))
plt.title('伽马变换后图像', fontdict={'family': 'KaiTi', 'size': 10})
cv2.imwrite('low_light_image4_4.jpg', img_gamma*255, [cv2.IMWRITE_PNG_COMPRESSION, 2])
plt.show()
