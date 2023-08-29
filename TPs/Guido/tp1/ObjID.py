# Ejercicios con la cámara, en tiempo real, mostrando siempre en una ventana la imagen de la cámara.  Requiere imágenes binarias,
# que se pueden obtener por thresholding.
# Dilatar una imagen binaria, controlar el tamaño del elemento estructural con un trackbar, de 1 a 50
# Erosionar
# Aplicar opening y closing consecutivamente, para filtrar ruidos
import math
import time

from PIL import Image
import cv2 as cv
import numpy as np

webcam = cv.VideoCapture(0)


def setBinary(image, val):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    ret1, thresh1 = cv.threshold(gray, val, 255,
                                 cv.THRESH_BINARY_INV)  # aplica funcion threadhole / ret1 si es true --> significa q no tenemos error
    return thresh1


def setBinaryAutom(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    ret1, thresh1 = cv.threshold(gray, 0, 255,
                                 cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # aplica funcion threadhole / ret1 si es true --> significa q no tenemos error
    return thresh1


def dilatation(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    return cv.dilate(img, kernel, iterations=1)


def erosion(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    return cv.erode(img, kernel, iterations=1)


def denoise(img, val1, val2):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (val1 + 1, val2 + 1))
    tempImg = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    return cv.morphologyEx(tempImg, cv.MORPH_CLOSE, kernel)


def getContours(binary, img):
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for i in contours:
        area = cv.contourArea(i)
        if area > 0.5:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri, True)
    return contours


def getBiggestContour(contours):
    try:
        max_cnt = contours[0]
        for cnt in contours:
            if cv.contourArea(cnt) > cv.contourArea(max_cnt):
                max_cnt = cnt
        return max_cnt
    except:
        return contours


def imagesContours():  # devuelve un array con todos los contornos de las img.
    circulo = setBinaryAutom(np.array(Image.open('circulo.png')))
    triangulo = setBinaryAutom(np.array(Image.open('triangulo.png')))
    rectangulo = setBinaryAutom(np.array(Image.open('rectangulo.png')))


    circuloContour, hierarchy = cv.findContours(circulo, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    trianguloContour, hierarchy = cv.findContours(triangulo, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    rectanguloContour, hierarchy = cv.findContours(rectangulo, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    contours = {
        "triangulo": getBiggestContour(trianguloContour),
        "circulo": getBiggestContour(circuloContour),
        "rectangulo": getBiggestContour(rectanguloContour)
    }
    return contours


def main():
    cv.namedWindow('binary')
    cv.createTrackbar('Thresh', 'binary', 0, 255, setBinary)
    cv.namedWindow('denoised')
    cv.createTrackbar('KSize', 'denoised', 0, 5, denoise)
    cv.namedWindow('webcam')
    cv.createTrackbar('Error', 'webcam', 0, 50, denoise)
    while True:
        tecla = cv.waitKey(30)
        ret, img = webcam.read()

        valBinary = cv.getTrackbarPos("Thresh", "binary")

        binaryImg = setBinary(img, valBinary)
        cv.imshow('binary', binaryImg)

        valKS = cv.getTrackbarPos('KSize', 'denoised')
        denoisedImg = denoise(binaryImg, valKS, valKS)
        cv.imshow('denoised', denoisedImg)

        contours1 = getContours(denoisedImg, img)

        valError = cv.getTrackbarPos("Error", 'webcam')
        for i in contours1:
            contours = imagesContours()
            for j in contours.keys():
                error = cv.matchShapes(i, contours[j], cv.CONTOURS_MATCH_I2, 0)
                if cv.contourArea(i) > 1000:
                    if error < valError/1000:
                        x, y, w, h = cv.boundingRect(i)
                        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)
                        cv.putText(img, j, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    else:
                        x, y, w, h = cv.boundingRect(i)
                        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv.imshow('webcam', img)
        if tecla == 27:
            break


main()
