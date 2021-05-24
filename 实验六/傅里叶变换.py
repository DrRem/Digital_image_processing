from math import e

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

im = cv.imread("89937761_p0_master1200.jpg")

im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)  # 将图片转为灰度图像

dft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im))))
# 傅里叶变换，将变换结果中心化，将变换结果取对数

img = np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(im)))))
# 傅里叶逆变换

plt.subplot(131), plt.imshow(im, 'gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(132), plt.imshow(dft, 'gray'), plt.title('Fourier Image')
plt.axis('off')
plt.subplot(133), plt.imshow(img, 'gray'), plt.title('Inverse Fourier Image')
plt.axis('off')

plt.show()