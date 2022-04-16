import cv2 as cv
import matplotlib.pyplot as plt

# %%
p1 = r"Y:\VIDEO\BEAUTYLEG_P\[Beautyleg]2021-05-26 No.11 ChiChi[1V462M]\11ChiChi.mp4"

video = cv.VideoCapture(p1)

while True:
    ret, img = video.read()
    cv.imshow('ChiChi.mp4', img)
    k = cv.waitKey(100)
    print(k)
    if k == ord('q'):
        exit()
