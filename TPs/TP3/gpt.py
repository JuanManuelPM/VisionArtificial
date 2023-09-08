import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Cargar la imagen de células
img = cv.imread('../../static/images/celulas.png')

# Convertir la imagen a escala de grises
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Aplicar umbral para separar el primer plano (núcleos) del fondo
_, binary_img = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

# Realizar dilatación y erosión para mejorar la segmentación
kernel_dilate = np.ones((33, 33), np.uint8)
kernel_erode = np.ones((3, 3), np.uint8)

# Dilatación del fondo
sure_bg = cv.dilate(binary_img, kernel_dilate, iterations=3)

# Erosión del primer plano
fg_eroded = cv.erode(binary_img, kernel_erode, iterations=1)

# Dilatación adicional del primer plano
kernel_fg_dilate = np.ones((20, 20), np.uint8)
fg_dilated = cv.dilate(fg_eroded, kernel_fg_dilate, iterations=1)

# Encontrar la región desconocida
unknown = cv.subtract(sure_bg, fg_dilated)

# Etiquetar componentes conectados en el primer plano
_, markers = cv.connectedComponents(fg_dilated)
markers = markers + 1
markers[unknown == 255] = 0

# Aplicar Watershed para segmentación
cv.watershed(img, markers)
for i in np.unique(markers):
    if i % 3 == 0:
        img[markers == i] = [255, 0, 0]
    elif i % 3 == 1:
        img[markers == i] = [0, 255, 0]
    else:
        img[markers == i] = [0, 0, 255]

# Colorear las células segmentadas en diferentes colores
img[markers == -1] = [0, 0, 255]

# Mostrar la imagen original con las células segmentadas
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.show()
