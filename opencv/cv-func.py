import cv2 as cv

# if __name__ == '__main__':
#     # file=r"F:\BaiduNetdiskDownload\[YOUMI]YMH20171111VOL0083 2017.11.11 VOL.083 妲己_Toxic\0016.jpg"
#
#
#     img = cv2.imread(file)
#     cv2.imshow('妲己', img)
#     cv2.waitKey(0)


import numpy as np
from matplotlib import pyplot as plt

print(cv.__version__)

file = r"E:\Resource\[Be]2019.03.15 No.1739 Nancy[58P306M]\0015.jpg"

img = cv.imread(file, 0)
ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
thresh6 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 2)
thresh7 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 9, 2)

print(ret)

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV', 'IMG', 'MEAN', 'GAUSSIAN']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5, img, thresh6, thresh7]

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
