"""
File: HSV_transform.py
Author: CUMT-Muzihao
Date: 2024/04/14
Description:
"""
import cv2
import numpy as np

# 读取图像
image = cv2.imread('../low_light_image4.jpg')

# 将图像从RGB空间转换到HSV空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 分割H、S、V通道
h, s, v = cv2.split(hsv_image)

# 对V通道进行直方图均衡化
equalized_v = cv2.equalizeHist(v)

# 将增强后的V通道与原始H、S通道合并回HSV图像
enhanced_hsv_image = cv2.merge((h, s, equalized_v))

# 将增强后的HSV图像转换回RGB空间
enhanced_image = cv2.cvtColor(enhanced_hsv_image, cv2.COLOR_HSV2BGR)

# 显示原始图像和增强后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Enhanced Image', enhanced_image)
cv2.imwrite('low_light_image4_3.jpg', enhanced_image, [cv2.IMWRITE_PNG_COMPRESSION, 2])
cv2.waitKey(0)
cv2.destroyAllWindows()