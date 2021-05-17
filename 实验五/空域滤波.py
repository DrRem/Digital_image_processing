from cv2 import cv2 as cv
import numpy as np
import random


def sp_noise(image, prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    # cv.imshow("gasuss", out)
    return out


if __name__ == '__main__':
    im = cv.imread('6.1.14.tiff')
    print(im.shape)

    im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)  # 将图片转为灰度图像

    im_gasuss = gasuss_noise(im, 0, 0.005)   # 添加高斯噪声

    im_sp = sp_noise(im, 0.1)    # 添加椒盐噪声

    cv.imshow('im', im)
    cv.imshow('gasuss', im_gasuss)
    cv.imshow('sp', im_sp)
    cv.waitKey(0)



    # blur3 = cv.blur(im, (3, 3))  # 平均滤波
    # blur5 = cv.blur(im, (5, 5))
    #
    # median3 = cv.medianBlur(im, 3)
    # median5 = cv.medianBlur(im, 5)
    #
    # cv.imshow('im', im)
    # cv.imshow('blur3', blur3)
    # cv.imshow('blur5', blur5)
    # cv.imshow('median3', median3)
    # cv.imshow('median5', median5)
    # cv.waitKey(0)
