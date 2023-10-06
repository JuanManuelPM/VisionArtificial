import numpy as np
import cv2 as cv

from joblib import load

# carga el modelo
clasificador = load('filename.joblib') 

cap = cv.VideoCapture(0)

th1 = 60
kernel = 0

def generarHu(listaHu):
    toR = []
    for i in listaHu:
        toR.append(i[0])

    return toR

def comparar(con1):
    moments_alphabet = cv.moments(con1)
    huMoments_alphabet = cv.HuMoments(moments_alphabet)
    etiquetaPredicha = clasificador.predict([generarHu(huMoments_alphabet)])
    return str(etiquetaPredicha)
    
def tresh(x):
    global th1
    th1 = x
    
def morph(aj):
    global kernel
    kernel = np.ones((aj, aj),np.uint8)
    
cv.namedWindow('th_inverted') #instancio una ventana
cv.createTrackbar('TBThresh', 'th_inverted', 0, 150, tresh)
cv.createTrackbar('TBKernel', 'th_inverted', 1, 20, morph)
    
codigo = {"1":"Avion", "2":"Circulo", "3":"Cuadrado", "4":"Estrella"}

font = cv.FONT_HERSHEY_SIMPLEX
  
# fontScale
fontScale = 1
   
# Red color in BGR
color = (0, 0, 255)
  
# Line thickness of 2 px
thickness = 2

th1 = 0


def get_biggest_contour(contours):
    try:
        max_cnt = contours[0]
        for cnt in contours:
            if cv.contourArea(cnt) > cv.contourArea(max_cnt):
                max_cnt = cnt
        return max_cnt
    except:
        return contours

while True:
    ret, frame = cap.read()
    frameOriginal= frame
    frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    ret1,img1 = cv.threshold(frame,th1, 255, cv.THRESH_BINARY_INV)
    img2 = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)

    contorno = img2

    contours, hierarchy = cv.findContours(img1, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        cv.drawContours(contorno, get_biggest_contour(contours), -1, (0, 255, 0), 3)

    comparacion= comparar(get_biggest_contour(contours))

    if(comparacion=="None"):
         for i in range(len(contours)):
            cv.drawContours(frameOriginal, get_biggest_contour(contours), -1, (0,0, 255), 3)
    else:
         for i in range(len(contours)):
            cv.drawContours(frameOriginal, get_biggest_contour(contours), -1, (0,255, 0), 3)   

    cv.imshow("Threadhol", img1)
    #cv.imshow("Kernel", img2)
    cv.putText(frameOriginal, codigo[comparacion[1]],(20, 20), font, 1, color, thickness, cv.LINE_AA)
    cv.imshow("Contorno", frameOriginal)

    cv.waitKey(1)
