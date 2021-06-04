import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

im = cv.imread('89937761_p0_master1200.jpg')
B, G, R = cv.split(im)

plt.subplot(331), plt.imshow(im, cmap='gray'), plt.title('Original'), plt.axis('off')
plt.subplot(334), plt.imshow(R, cmap='gray'), plt.title('R'), plt.axis('off')
plt.subplot(335), plt.imshow(G, cmap='gray'), plt.title('G'), plt.axis('off')
plt.subplot(336), plt.imshow(B, cmap='gray'), plt.title('B'), plt.axis('off')

plt.subplot(337), plt.hist(im.ravel(), 256, [0, 256], color='r'), plt.axis('off')
plt.subplot(338), plt.hist(im.ravel(), 256, [0, 256], color='g'), plt.axis('off')
plt.subplot(339), plt.hist(im.ravel(), 256, [0, 256], color='b'), plt.axis('off')

img = cv.cvtColor(im, cv.COLOR_RGB2GRAY)

img = cv.equalizeHist(img)
plt.subplot(332), plt.imshow(img, cmap='gray'), plt.title('equalizeHist'), plt.axis('off')

plt.show()