import cv2 as cv

image_circle = cv.imread('../../static/images/circulo.jpg')
image_triangle = cv.imread('../../static/images/triangulo.jpg')
image_square = cv.imread('../../static/images/cuadrado.jpg')

gray = cv.cvtColor(image_circle, cv.COLOR_BGR2GRAY)
# Any sharp edges in images are smoothed while minimizing too much blurring.
blurred_gray = cv.GaussianBlur(gray, (5, 5), 0)
# We also used THRESH_OTSU to analyze the image and determine the threshold.
ret3, thresh = cv.threshold(blurred_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cv.drawContours(image_circle, contours, -1, (0, 0, 255), 2)
cv.imwrite('image.jpg', image_circle)
