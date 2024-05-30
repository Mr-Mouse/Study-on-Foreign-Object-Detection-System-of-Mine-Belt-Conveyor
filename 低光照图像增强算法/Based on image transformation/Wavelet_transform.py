"""
File: Wavelet_transform.py
Author: CUMT-Muzihao
Date: 2024/04/12
Description:小波变换实现图像亮度增强
"""
import cv2
import numpy as np
import pywt
from matplotlib import pyplot as plt

# 加载图像
image = cv2.imread('../low_light_image4.jpg', cv2.IMREAD_GRAYSCALE)

# 小波变换
coeffs = pywt.dwt2(image, 'haar')
cA, (cH, cV, cD) = coeffs

# 增强图像的对比度，可以调整增强系数来控制效果
alpha = 2.5
cA_enhanced = cA * alpha

# 逆小波变换
coeffs_enhanced = cA_enhanced, (cH, cV, cD)
image_enhanced = pywt.idwt2(coeffs_enhanced, 'haar')

# 结果显示
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('原图像', fontdict={'family': 'KaiTi', 'size': 10}), plt.axis('off')
plt.subplot(122), plt.imshow(np.uint8(image_enhanced), cmap='gray')
plt.title('增强后图像', fontdict={'family': 'KaiTi', 'size': 10}), plt.axis('off')
cv2.imwrite('low_light_image4_5.jpg', np.uint8(image_enhanced), [cv2.IMWRITE_PNG_COMPRESSION, 2])
plt.show()
