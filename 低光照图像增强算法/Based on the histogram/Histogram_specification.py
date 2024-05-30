"""
File: Histogram_specification.py
Author: CUMT-Muzihao
Date: 2024/04/12
Description:直方图规定化亮度增强
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread('../low_light_image1.jpg')  # 读入原图像
img2 = cv2.imread('../Reference_picture.jpg') # 读入参考图像

plt.subplot(2, 3, 1)
plt.imshow(img1, cmap='gray')
plt.axis('off')
plt.title('原图像', fontdict={'family': 'KaiTi', 'size': 10})

plt.subplot(2, 3, 2)
plt.imshow(img2, cmap='gray')
plt.axis('off')
plt.title('参考图像', fontdict={'family': 'KaiTi', 'size': 10})

# 计算原图像和参考图像的直方图
hist1, bins1 = np.histogram(img1.flatten(), 256, [0, 256])
hist2, bins2 = np.histogram(img2.flatten(), 256, [0, 256])

# 将直方图归一化
hist1 = hist1 / float(np.sum(hist1))
hist2 = hist2 / float(np.sum(hist2))

# 计算原图像和参考图像的累积分布函数（CDF）
cdf1 = hist1.cumsum()
cdf2 = hist2.cumsum()

# 将CDF归一化
cdf1 = cdf1 / float(cdf1[-1])
cdf2 = cdf2 / float(cdf2[-1])

# 创建新的图像数组
img3 = np.zeros_like(img1)

# 计算灰度值映射
lut = np.interp(cdf1, cdf2, np.arange(0, 256))

# 针对每个像素应用灰度映射
for i in range(256):
    img3[img1 == i] = lut[i]

# 显示规定化后的图像
plt.subplot(2, 3, 3)
plt.imshow(img3, cmap='gray')
plt.axis('off')
plt.title('规定化后的图像', fontdict={'family': 'KaiTi', 'size': 10})
cv2.imwrite('low_light_image1_2.jpg', img3, [cv2.IMWRITE_PNG_COMPRESSION, 2])
B, _, _ = cv2.split(img1)  # get single 8-bits channel
b = cv2.equalizeHist(B)

hist_B = cv2.calcHist([img1], [0], None, [256], [0, 256])
hist_B_ = cv2.calcHist([img2], [0], None, [256], [0, 256])
hist_b = cv2.calcHist([img3], [0], None, [256], [0, 256])


plt.subplot(2, 3, 4)
plt.plot(hist_B, 'b')
plt.title('原图B通道直方图', fontdict={'family': 'KaiTi', 'size': 10})
plt.subplot(2, 3, 5)
plt.plot(hist_B_, 'b')
plt.title('参考图B通道直方图', fontdict={'family': 'KaiTi', 'size': 10})
plt.subplot(2, 3, 6)
plt.title('规定化后B通道直方图', fontdict={'family': 'KaiTi', 'size': 10})
plt.plot(hist_b, 'b')
plt.show()