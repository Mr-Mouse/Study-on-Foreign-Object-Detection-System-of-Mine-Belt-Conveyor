"""
File: Histogram_equalization.py
Author: CUMT-Muzihao
Date: 2024/04/12
Description:直方图均衡化亮度增强
"""
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("../resuorse/low_light_image4.jpg")
B, G, R = cv2.split(img)  # get single 8-bits channel
b = cv2.equalizeHist(B)
g = cv2.equalizeHist(G)
r = cv2.equalizeHist(R)
equal_img = cv2.merge((b, g, r))  # merge it back

hist_B = cv2.calcHist([img], [0], None, [256], [0, 256])

hist_b = cv2.calcHist([equal_img], [0], None, [256], [0, 256])

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('原图像', fontdict={'family': 'KaiTi', 'size': 10})

plt.subplot(2, 2, 2)
plt.imshow(equal_img, cmap='gray')
plt.axis('off')
plt.title('均衡化后的图像', fontdict={'family': 'KaiTi', 'size': 10})

plt.subplot(2, 2, 3)
plt.plot(hist_B, 'b')
plt.title('原图B通道直方图', fontdict={'family': 'KaiTi', 'size': 10})

plt.subplot(2, 2, 4)
plt.title('均衡化后B通道直方图', fontdict={'family': 'KaiTi', 'size': 10})
plt.plot(hist_b, 'b')
plt.show()

cv2.imshow("orj", img)
cv2.imshow("equal_img", equal_img)
cv2.imwrite('low_light_image4_1.jpg', equal_img, [cv2.IMWRITE_PNG_COMPRESSION, 2])
cv2.waitKey(0)
