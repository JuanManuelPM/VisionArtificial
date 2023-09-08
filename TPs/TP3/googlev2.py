# Importamos las dependencias
from argparse import ArgumentParser

import cv2
import numpy as np

# Definimos el menú del programa
argument_parser = ArgumentParser()
#argument_parser.add_argument('-i', '--image', required=True, help='Ruta a la imagen de entrada.')
arguments = vars(argument_parser.parse_args())

# Cargamos la imagen
image = cv2.imread('../../static/images/monedas.jpeg')

# Aplicamos el umbral de OTSU para hallar una estimación aproximada de los objetos en la imagen.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow('Binaria', thresholded)

# Remueve ruido.
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)

# Dilatamos para obtener las regiones que estamos seguros que pertenecen al fondo
background = cv2.dilate(opening, kernel, iterations=3)
cv2.imshow('Fondo', background)

# Aplicamos distanceTransform para hallar las regiones que estamos seguros que pertenecen al primer plano.
distance_transform = cv2.distanceTransform(opening, cv2.DIST_L2, maskSize=5)
_, foreground = cv2.threshold(distance_transform, 0.7 * distance_transform.max(), 255, 0)
foreground = np.uint8(foreground)
cv2.imshow('Primer plano', foreground)

# Ahora hallamos las regiones sobre las que no estamos del todo seguros
unknown = cv2.subtract(background, foreground)
cv2.imshow('Desconocido', unknown)

# Hallamos las componentes conectadas
_, markers = cv2.connectedComponents(foreground)

# Para que watershed no considere el fondo como una región desconocida, tenemos que etiquetarla con un valor distinto
# a 0, por lo que sumamos uno a los markers
markers = markers + 1

# Ahora sí, usamos el cero para denotar las regiones verdaderamente desconocidas.
markers[unknown == 255] = 0

# Aplicamos watershed
markers = cv2.watershed(image, markers)
for i in np.unique(markers):
    if i % 3 == 0:
        image[markers == i] = [255, 0, 0]
    elif i % 3 == 1:
        image[markers == i] = [0, 255, 0]
    else:
        image[markers == i] = [0, 0, 255]
image[markers == -1] = [0, 0, 255]

# Mostramos los objetos segmentados.
cv2.imshow('Marcadores', image)
cv2.waitKey(0)