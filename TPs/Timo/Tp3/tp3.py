import cv2
import numpy as np
from matplotlib import pyplot as pl

k = 20

kernel = np.ones((12, 12),np.uint8)
kernel_fg = np.ones((k , k),np.uint8)
kernel_bg = np.ones((22, 22),np.uint8)

img = cv2.imread(".\\levadura.png")
imgOriginal = cv2.imread(".\\levadura.png")
im_Gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# cv2.imshow("gray", im_Gray)
# cv2.waitKey(0)


ret1, fg = cv2.threshold(im_Gray, 100,255,cv2.THRESH_BINARY)
ret2, bg = cv2.threshold(im_Gray, 40,255,cv2.THRESH_BINARY)
ret2, bg_unk = cv2.threshold(im_Gray, 40,255,cv2.THRESH_BINARY)


fg_erode = cv2.morphologyEx(fg, cv2.MORPH_ERODE, kernel)
fg_dilate = cv2.morphologyEx(fg, cv2.MORPH_DILATE, kernel_fg)
bg_dilate = cv2.morphologyEx(bg, cv2.MORPH_DILATE, kernel_bg)

cv2.imshow("fg_dilate", fg_dilate)
# cv2.imshow("bg", bg)
cv2.imshow("bg_dilate", bg_dilate)
cv2.waitKey(0)

unk = cv2.subtract(bg_dilate,fg_dilate)

# cv2.imshow("unk", unk)
# cv2.waitKey(0)

bg_dilate = cv2.bitwise_not(bg_dilate)

# Marker labelling
_, markers = cv2.connectedComponents(fg_dilate)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unk == 255] = 0

pl.imshow(markers, cmap="jet")
pl.show()

markers = cv2.watershed(img, markers)

for i in np.unique(markers):
   if i%3 == 0:
       img[markers==i] = [255,0,0]
   elif i%3 == 1:
       img[markers==i] = [0,255,0]
   else:
       img[markers==i] = [0,0,255]

#img[markers==-1] = [0,255,0]
img[markers==1] = [255,255,0]
# img[markers==22] = [255,255,255]

cv2.imshow("original", imgOriginal)

pl.imshow(img, cmap="jet")
pl.show()

