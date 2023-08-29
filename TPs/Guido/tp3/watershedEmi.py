import cv2 as cv
import numpy as np
from matplotlib import pyplot as pl

def setBinary(image, val):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    ret1, fg = cv.threshold(gray, val, 255,cv.THRESH_BINARY)  # aplica funcion threadhole / ret1 si es true --> significa q no tenemos error
    ret2, bg = cv.threshold(gray, val, 255,cv.THRESH_BINARY)  # aplica funcion threadhole / ret1 si es true --> significa q no tenemos error
    return fg , bg

def lam(x):
    return x


def main():
    cv.namedWindow('Tracks')
    cv.createTrackbar('Thresh', 'Tracks', 73, 255, lam)
    cv.createTrackbar('Dilate', 'Tracks', 33, 100, lam)
    cv.createTrackbar('Erode', 'Tracks', 3, 100, lam)
    cv.createTrackbar('DilateFg', 'Tracks', 20, 100, lam)

    while True:
        key = cv.waitKey(30)

        #ToBinary
        threshVal = cv.getTrackbarPos('Thresh', 'Tracks')
        img = cv.imread("../../../../OneDrive/Escritorio/visionArtificial/tp3/levadura.png")
        fg , bg = setBinary(img, threshVal)
        #cv.imshow("bin", imgBinary)


        # sure background area
        dilateVal = cv.getTrackbarPos('Dilate', 'Tracks')
        kernel = np.ones((dilateVal, dilateVal), np.uint8)
        sure_bg = cv.dilate(bg, kernel, iterations=3)

        # Finding sure foreground area
        erodeVal = cv.getTrackbarPos('Erode', 'Tracks')
        kernelErode = np.ones((erodeVal, erodeVal), np.uint8)
        fg_eroded = cv.morphologyEx(fg, cv.MORPH_ERODE, kernelErode)

        dilateVal2 = cv.getTrackbarPos('DilateFg', 'Tracks')
        kernelFgDilation = np.ones((dilateVal2, dilateVal2), np.uint8)
        fgDilated = cv.morphologyEx(fg_eroded,cv.MORPH_DILATE, kernelFgDilation)

        # dist_transform = cv.distanceTransform(fg_eroded, cv.DIST_L2, 5)
        #ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        unknown = cv.subtract(sure_bg, fgDilated)
        #cv.imshow("sureFg", fgDilated)
        #cv.imshow("subreBg", sure_bg)

        #Markers
        _, markers = cv.connectedComponents(fgDilated)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        pl.imshow(markers, cmap="jet")
        pl.show()

        markers = cv.watershed(img, markers)
        for i in np.unique(markers):
            if i % 3 == 0:
                img[markers == i] = [255, 0, 0]
            elif i % 3 == 1:
                img[markers == i] = [0, 255, 0]
            else:
                img[markers == i] = [0, 0, 255]

        # img[markers==-1] = [0,255,0]
        img[markers == 1] = [255, 255, 0]
        # img[markers==22] = [255,255,255]

        cv.imshow("original", img)

        pl.imshow(img, cmap="jet")
        pl.show()

        if key == 27:
            break

main()

