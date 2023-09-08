import cv2 as cv


def binary(val):
    image = cv.imread('../../static/images/celulas.png')
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret1, thresh1 = cv.threshold(gray, val, 255, cv.THRESH_BINARY)
    cv.imshow("Binary", thresh1)

alpha_slider_max = 100

cv.namedWindow('Binary')
cv.createTrackbar('Trackbar', 'Binary', 0, alpha_slider_max, binary)

# Show some stuff
binary(0)
# Wait until user press some key
cv.waitKey()

# sure background area
dilateVal = cv.getTrackbarPos('Dilate', 'Tracks')
kernel = np.ones((dilateVal, dilateVal), np.uint8)
sure_bg = cv.dilate(bg, kernel, iterations=3)


