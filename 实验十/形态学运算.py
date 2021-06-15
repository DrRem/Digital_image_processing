import cv2 as cv
import numpy as np

im = cv.imread('89937761_p0_master1200.jpg', cv.IMREAD_GRAYSCALE)
im = cv.resize(im, (256, 256))
cv.imshow('s', im)

# 设置卷积核
kernel = np.ones((5, 5), np.uint8)

# 开运算
erode_im = cv.erode(im, kernel)  # 腐蚀
dilate_result = cv.dilate(erode_im, kernel)  # 膨胀

# 闭运算
dilate_result_1 = cv.dilate(im, kernel)
erode_im_1 = cv.erode(dilate_result_1, kernel)

cv.imshow('kai', dilate_result)
cv.imshow('bi', erode_im_1)
cv.waitKey(0)