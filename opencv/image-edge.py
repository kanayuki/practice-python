import cv2 as cv
import sys
from matplotlib import pyplot as plt

# %%
path = r"Y:\IMAGE\BEAUTYLEG\[Be]2021.05.19 No.2079 ChiChi[47P562M]\0016.jpg"
img = cv.imread(path)

if img is None:
    sys.exit("Could not read the image.")

plt.imshow(img)

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img)

plt.show()

# %%
print(img.shape)

# %%
img_new = cv.resize(img, (512, 512))
print(img_new.shape)

# %%
plt.imshow(img_new)
plt.show()

# %%
img_crop = img[1500:2500, 1500:2500]
plt.matshow(img_crop)
plt.show()

# %%
img_gray = cv.cvtColor(img_crop, cv.COLOR_RGB2GRAY)
plt.matshow(img_gray)
plt.show()

# %%
img_blur = cv.GaussianBlur(img_crop, (3, 3), 2)
plt.matshow(img_blur)
plt.show()

#%%
img_edge = cv.Canny(img_crop, 150, 250)
plt.matshow(img_edge)
plt.show()


#%%
img_dilate = cv.dilate(img_edge, (5,5))
plt.matshow(img_dilate)
plt.show()
