import cv2 as cv
import numpy as np
from pywt import dwt2, idwt2

im = cv.imread('sample.jpg', 0)
# 以灰度模式读取图像

im = cv.resize(im, (512, 512))

cv.imshow('im', im)

# 对img进行haar小波变换：
cA, (cH, cV, cD) = dwt2(im, 'haar')
 
# 小波变换之后，低频分量对应的图像：
im_low = np.uint8(cA/np.max(cA)*255)
# 小波变换之后，水平方向高频分量对应的图像：
im_high = np.uint8(cH/np.max(cH)*255)
# 小波变换之后，垂直平方向高频分量对应的图像：
im_v = np.uint8(cV/np.max(cV)*255)
# 小波变换之后，对角线方向高频分量对应的图像：
im_d = np.uint8(cD/np.max(cD)*255) 
# 根据小波系数重构回去的图像
rim = idwt2((cA,(cH,cV,cD)), 'haar')
rim = np.uint8(rim)

cv.imshow('high', im_high)
cv.imshow('low', im_low)
cv.imshow('v', im_v)
cv.imshow('d', im_d)
cv.imshow('rim', rim)
cv.waitKey(0)