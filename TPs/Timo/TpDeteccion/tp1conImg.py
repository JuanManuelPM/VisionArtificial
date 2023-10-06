import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

th1 = 60
kernel = 0
th2= 0.3

circulo = cv.imread("circulo.jpeg")
cuadrado = cv.imread("cuadrado.jpeg")
estrella = cv.imread("estrella.jpeg")
avion = cv.imread("avion.jpeg")

circulo = cv.cvtColor(circulo, cv.COLOR_RGB2GRAY)
cuadrado = cv.cvtColor(cuadrado, cv.COLOR_RGB2GRAY)
estrella = cv.cvtColor(estrella, cv.COLOR_RGB2GRAY)
avion = cv.cvtColor(avion, cv.COLOR_RGB2GRAY)

ret1,circulo = cv.threshold(circulo,th1, 255, cv.THRESH_BINARY_INV)
ret1,cuadrado = cv.threshold(cuadrado,th1, 255, cv.THRESH_BINARY_INV)
ret1,estrella = cv.threshold(estrella,th1, 255, cv.THRESH_BINARY_INV)
ret1,avion = cv.threshold(avion,th1, 255, cv.THRESH_BINARY_INV)


circuloCon, hierarchy = cv.findContours(circulo, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cuadradoCon, hierarchy = cv.findContours(cuadrado, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
estrellaCon, hierarchy = cv.findContours(estrella, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
avionCon,   hierarchy = cv.findContours(avion, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

def get_biggest_contour(contours):
    try:
        max_cnt = contours[0]
        for cnt in contours:
            if cv.contourArea(cnt) > cv.contourArea(max_cnt):
                max_cnt = cnt
        return max_cnt
    except:
        return contours

allCon = {
    "cuadrado":get_biggest_contour(cuadradoCon),
    "estrella": get_biggest_contour(estrellaCon),
    "circulo":get_biggest_contour(circuloCon),
    "avion":get_biggest_contour(avionCon)
}


def tresh(x):
    global th1
    th1 = x
    
def morph(aj):
    global kernel
    kernel = np.ones((aj, aj),np.uint8)

def sensitivity(x):
    global th2
    th2 = x/10
    
cv.namedWindow('th_inverted') #instancio una ventana
cv.createTrackbar('TBThresh', 'th_inverted', 0, 150, tresh)
cv.createTrackbar('TBKernel', 'th_inverted', 1, 20, morph)
cv.createTrackbar("TBSensitivity", "th_inverted", 0,10, sensitivity)
    

def comparar(con1):
    moments_alphabet = cv.moments(con1)
    huMoments_alphabet = cv.HuMoments(moments_alphabet)
    for k in allCon.keys():
        if cv.matchShapes(con1, allCon[k], cv.CONTOURS_MATCH_I2, 0) < th2:
            return k
    return "None"

font = cv.FONT_HERSHEY_SIMPLEX
  
# fontScale
fontScale = 1
   
# Red color in BGR
color = (0, 0, 255)
  
# Line thickness of 2 px
thickness = 2

th1 = 0

while True:
    ret, frame = cap.read()
    frameOriginal= frame
    frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    ret1,img1 = cv.threshold(frame,th1, 255, cv.THRESH_BINARY_INV)
    img2 = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)

    contorno = img2

    contours, hierarchy = cv.findContours(img1, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
   

    cv.imshow("Threadhol", img1)
    # cv.imshow("Kernel", img2)
    comparacion= comparar(get_biggest_contour(contours))
    if(comparacion=="None"):
         for i in range(len(contours)):
            cv.drawContours(frameOriginal, get_biggest_contour(contours), -1, (0,0, 255), 3)
    else:
         for i in range(len(contours)):
            cv.drawContours(frameOriginal, get_biggest_contour(contours), -1, (0,255, 0), 3)       
    cv.putText(frameOriginal, comparacion,(20, 20), font, 1, color, thickness, cv.LINE_AA)
    cv.imshow("Contorno", frameOriginal)

    cv.waitKey(1)