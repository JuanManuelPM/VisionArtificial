import cv2
import cv2 as cv

#generar la imagen normal
image = cv.imread('../../static/images/circulo.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)

#cv.imshow('Circulo', gray)
#cv.waitKey(0)

#Agregar threshold.
#threshold(imagen en escala de grises, valor de umbral, valor máximo que se le asigna a los pixeles que exceden el umbral, tipo de threshold)
def binary_inv(val):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    ret2, thresh2 = cv.threshold(gray, val, 255, cv.THRESH_BINARY_INV)
    cv.imshow("BinaryInv", thresh2)

cv.namedWindow('BinaryInv')
cv.createTrackbar('Trackbar', 'BinaryInv', 0, 255, binary_inv)
binary_inv(0)
cv.waitKey(0)
