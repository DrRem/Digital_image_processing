from cv2 import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == '__main__':
    im = cv.imread('7.2.01.tiff')
    im = cv.resize(im, (265, 265))
    im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)  # 将图片转为灰度图像
    plt.hist(im.flatten(), 256)  # 使用plt生成直方图

    im1 = cv.equalizeHist(im)  # 对原图进行直方图均衡化

    plt.hist(im1.flatten(), 256)  # 使用plt生成直方图
    plt.show()  # 使用matplotlib显示直方图

    cv.imshow('before', im)
    cv.imshow('after', im1)
    cv.waitKey(0)