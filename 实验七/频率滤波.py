import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def lowPassFilter(image, d):    # 低通滤波器
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    def make_transform_matrix(d):
        transfor_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x: (x - 1) / 2, (800, 800)))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa, pb):
                    from math import sqrt
                    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                    return dis

                dis = cal_distance(center_point, (i, j))
                if dis <= d:
                    transfor_matrix[i, j] = 1
                else:
                    transfor_matrix[i, j] = 0
        return transfor_matrix

    d_matrix = make_transform_matrix(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
    return new_img


def highPassFilter(image, d):  # 高通滤波器
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    def make_transform_matrix(d):
        transfor_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x: (x - 1) / 2, (800, 800)))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa, pb):
                    from math import sqrt
                    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                    return dis

                dis = cal_distance(center_point, (i, j))
                if dis <= d:
                    transfor_matrix[i, j] = 0
                else:
                    transfor_matrix[i, j] = 1
        return transfor_matrix

    d_matrix = make_transform_matrix(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
    return new_img


im = cv.imread("89937761_p0_master1200.jpg")

im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
plt.subplot(151), plt.imshow(im, 'gray'), plt.title('Original')
plt.axis("off")

ldft = lowPassFilter(im, 100)
plt.subplot(152), plt.imshow(ldft, 'gray'), plt.title('LPF')
plt.axis("off")

mid = cv.medianBlur(im, 5)
plt.subplot(153), plt.imshow(mid, 'gray'), plt.title('5*5median')
plt.axis("off")

sub = cv.absdiff(np.int32(ldft), np.int32(mid))
plt.subplot(154), plt.imshow(sub, 'gray'), plt.title('sub')
plt.axis("off")

hdft = highPassFilter(im, 100)
plt.subplot(155), plt.imshow(hdft, 'gray'), plt.title('HPF')
plt.axis("off")

plt.show()
