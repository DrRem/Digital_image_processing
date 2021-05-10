from cv2 import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == '__main__':
    im = cv.imread('7.2.01.tiff')
    im = cv.resize(im, (256, 256))

    im1 = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
    hist = cv.calcHist(im1, [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.show()

    rows = im1.shape[0]
    cols = im1.shape[1]
    for i in range(rows):
        for j in range(cols):
            im1[i][j] = 255 - im1[i][j]

    hist = cv.calcHist(im1, [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.show()
    cv.imshow('im1', im1)

    mod = cv.getRotationMatrix2D((128, 128), 60, 1)
    im2 = cv.warpAffine(im, mod, (256, 256))
    cv.imshow('im2', im2)

    im3 = cv.flip(im, 1)  # 水平翻转
    cv.imshow('im3', im3)
    im4 = cv.flip(im, 0)  # 垂直翻转
    cv.imshow('im4', im4)
    im5 = cv.flip(im, -1)  # 镜像翻转
    cv.imshow('im5', im5)
    cv.waitKey(0)
