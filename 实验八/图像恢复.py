import math

import cv2 as cv
import numpy as np
import numpy.fft as fft
import random
import matplotlib.pyplot as plt


def sp_noise(image, prob):
    """
    添加椒盐噪声
    prob:噪声比例
    """
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
    """
        添加高斯噪声
        mean : 均值
        var : 方差
    """
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


def lowPassFilter(image, d):  # 低通滤波器
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


def AdaptProcess(src, i, j, minSize, maxSize):
    filter_size = minSize

    kernelSize = filter_size // 2
    rio = src[i - kernelSize:i + kernelSize + 1, j - kernelSize:j + kernelSize + 1]
    minPix = np.min(rio)
    maxPix = np.max(rio)
    medPix = np.median(rio)
    zxy = src[i, j]

    if (medPix > minPix) and (medPix < maxPix):
        if (zxy > minPix) and (zxy < maxPix):
            return zxy
        else:
            return medPix
    else:
        filter_size = filter_size + 2
        if filter_size <= maxSize:
            return AdaptProcess(src, i, j, filter_size, maxSize)
        else:
            return medPix


def adapt_meadian_filter(img, minsize, maxsize):  # 自适应中值滤波
    borderSize = maxsize // 2

    src = cv.copyMakeBorder(img, borderSize, borderSize, borderSize, borderSize, cv.BORDER_REFLECT)

    for m in range(borderSize, src.shape[0] - borderSize):
        for n in range(borderSize, src.shape[1] - borderSize):
            src[m, n] = AdaptProcess(src, m, n, minsize, maxsize)

    dst = src[borderSize:borderSize + img.shape[0], borderSize:borderSize + img.shape[1]]
    return dst


def motion_process(image_size, motion_angle):
    PSF = np.zeros(image_size)
    print(image_size)
    center_position = (image_size[0] - 1) / 2
    print(center_position)

    slope_tan = math.tan(motion_angle * math.pi / 180)
    slope_cot = 1 / slope_tan
    if slope_tan <= 1:
        for i in range(15):
            offset = round(i * slope_tan)  # ((center_position-i)*slope_tan)
            PSF[int(center_position + offset), int(center_position - offset)] = 1
        return PSF / PSF.sum()  # 对点扩散函数进行归一化亮度
    else:
        for i in range(15):
            offset = round(i * slope_cot)
            PSF[int(center_position - offset), int(center_position + offset)] = 1
        return PSF / PSF.sum()


def make_blurred(src, psf, eps):    # 对图片进行运动模糊
    input_fft = fft.fft2(src)  # 进行二维数组的傅里叶变换
    PSF_fft = fft.fft2(psf) + eps
    blurred = fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred


def wiener(src, psf, eps, k=0.01):  # 维纳滤波，K=0.01
    input_fft = fft.fft2(src)
    PSF_fft = fft.fft2(psf) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + k)
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))
    return result


im = cv.imread("89937761_p0_master1200.jpg")

im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
plt.subplot(331), plt.imshow(im, 'gray'), plt.title('Original'), plt.axis("off")

im_sp = sp_noise(im, 0.1)
im_gas = gasuss_noise(im, 0, 0.05)
plt.subplot(332), plt.imshow(im_sp, 'gray'), plt.title('sp'), plt.axis("off")
plt.subplot(333), plt.imshow(im_gas, 'gray'), plt.title('gas'), plt.axis("off")

im_sp_adapt = adapt_meadian_filter(im_sp, 3, 9)
im_gas_adapt = adapt_meadian_filter(im_gas, 3, 9)
plt.subplot(335), plt.imshow(im_sp_adapt, 'gray'), plt.title('spa'), plt.axis("off")
plt.subplot(336), plt.imshow(im_gas_adapt, 'gray'), plt.title('gasa'), plt.axis("off")

dsf = motion_process(im.shape, 45)
im_mot = np.abs(make_blurred(im, dsf, 1e-3))
plt.subplot(338), plt.imshow(im_mot, 'gray'), plt.title('motion'), plt.axis("off")

im_wiener = wiener(im_mot, dsf, 1e-3)
plt.subplot(339), plt.imshow(im_wiener, 'gray'), plt.title('wiener'), plt.axis("off")

plt.show()
