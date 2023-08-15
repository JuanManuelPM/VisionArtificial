import cv2 as cv

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
cv.createTrackBar('Trackbar', 'Operaciones Morfológicas', 0, 150, denoise)