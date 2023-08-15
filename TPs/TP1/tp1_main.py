import cv2
import cv2 as cv

# generar la imagen normal
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


#Agregar threshold.
#threshold(imagen en escala de grises, valor de umbral, valor máximo que se le asigna a los pixeles que exceden el umbral, tipo de threshold)
def binary_inv(val):
    gray = cv.cvtColor(image_circle, cv.COLOR_RGB2GRAY)
    ret2, thresh2 = cv.threshold(gray, val, 255, cv.THRESH_BINARY_INV)
    cv.imshow("BinaryInv", thresh2)

cv.namedWindow('BinaryInv')
cv.createTrackbar('Trackbar', 'BinaryInv', 0, 255, binary_inv)
binary_inv(0)
cv.waitKey(0)


#Para eliminar el ruido de la imagen, es necesario erosionar y dilatar la imagen.
#La dilatación nos va a permitir distinguir el fondo, mientras que la erosión nos permite separar objetos y eliminar detalles finos pequeños
#Se trabaja con imagenes binarias
#Open elimina puntos blancos (1. erosion, 2. dilatación)
#Close elimina puntos negros (1. dilatación, 2. erosion)
#El orden dilatación-erosion depende de quien genere más ruido (va a ser el primero)
#Kernel
#Radio tiene que ser impar y positivo

def denoise(frame, method, radius):
    kernel = cv.getStructuringElement(method, (radius, radius))
    opening = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    return closing


cv.namedWindow("Operaciones Morfológicas")
cv.createTrackBar('Trackbar', 'Operaciones Morfológicas', 0, 20, denoise)
cv.createTrackbar('Area contorno minima', 'Operaciones Morfológicas', 0, 100,  )


#CHAIN_APPROX_NONE registra un punto por pixel
#RETR_TREE genera una jerarquia de arbol, con contornos dentro de contornos

#al presionar "espacio" se captura el frame actual para generar una imagen binaria la cual podremos ajustar.
frame = 0
while cv.waitKey(1) != ord(' '):
    # iniciamos la capturadora con el nombre cap
    cap = cv.VideoCapture(0)
    # metodo para que la camara "lea"
    # frame son las "imagenes" (frames) de cada milisegundo
    ret, frame = cap.read()
    # aca las espejamos para que se vean bien
    frame = cv.flip(frame, 1)
    # aca las mostramos en una ventana

    def binary_inv(val):
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        ret2, thresh2 = cv.threshold(gray, val, 255, cv.THRESH_BINARY_INV)
        cv.imshow("BinaryInv", thresh2)

    cv.imshow('img1', frame)

