import cv2
import numpy as np

# Carga de la imagen de células
image = cv2.imread('../../static/images/celulas.png')

# Convierte la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Define la función de umbralización
def update_threshold(threshold_value):
    _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Encuentra y cuenta las células segmentadas
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cell_count = len(contours)

    # Dibuja los contornos de las células en la imagen original
    result_image = image.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

    # Muestra la cantidad de células encontradas en la imagen
    cv2.putText(result_image, f'Cells: {cell_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Muestra la imagen resultante
    cv2.imshow('Segmented Cells', result_image)

    return thresholded


# Crea una ventana con trackbar para ajustar el umbral
cv2.namedWindow('Segmented Cells')
cv2.createTrackbar('Threshold', 'Segmented Cells', 0, 255, update_threshold)

# Inicializa la ventana con un valor de umbral por defecto
initial_threshold = 128
cv2.setTrackbarPos('Threshold', 'Segmented Cells', initial_threshold)

# Llama a la función de umbralización inicial y obtiene la imagen binarizada
thresholded_image = update_threshold(initial_threshold)

# Espera hasta que se presione la tecla 'Esc' para continuar
while True:
    key = cv2.waitKey(10)
    if key == 27:  # 27 es el código ASCII para la tecla 'Esc'
        break

# Cierra la ventana actual
cv2.destroyWindow('Segmented Cells')

# Paso 2: Individualizar con componentes conectados
# a) Produce un mapa de semillas
distance_transform = cv2.distanceTransform(thresholded_image, cv2.DIST_L2, maskSize=5)
_, foreground = cv2.threshold(distance_transform, 0.7 * distance_transform.max(), 255, 0)
foreground = np.uint8(foreground)

# b) Visualiza el mapa de semillas con colormap
cv2.imshow('Seeds Map', cv2.applyColorMap(foreground, cv2.COLORMAP_JET))

# Espera hasta que se presione la tecla 'Esc' para continuar
while True:
    key = cv2.waitKey(10)
    if key == 27:
        break

# Cierra la ventana actual
cv2.destroyWindow('Seeds Map')


# Paso 3: Agregar al menos una semilla para el fondo
# a) Genera un mapa de semilla de fondo con otro threshold
def update_background_threshold(background_threshold_value):
    _, background_seeds = cv2.threshold(gray, background_threshold_value, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('Background Seeds', background_seeds)


# Crea una ventana con trackbar para ajustar el umbral del fondo
cv2.namedWindow('Background Seeds')
cv2.createTrackbar('Background Threshold', 'Background Seeds', 0, 255, update_background_threshold)

# Inicializa la ventana con un valor de umbral por defecto
initial_background_threshold = 100
cv2.setTrackbarPos('Background Threshold', 'Background Seeds', initial_background_threshold)

# Llama a la función de umbralización de fondo inicial
update_background_threshold(initial_background_threshold)

# Espera hasta que se presione la tecla 'Esc' para continuar
while True:
    key = cv2.waitKey(10)
    if key == 27:
        break

# Cierra la ventana actual
cv2.destroyWindow('Background Seeds')

# Paso 4: Ejecutar watershed para producir la segmentación
_, markers = cv2.connectedComponents(foreground)
markers = markers + 1

# Obtén los marcadores de fondo (background_seeds) a partir del umbral de fondo
_, background_seeds = cv2.threshold(gray, initial_background_threshold, 255, cv2.THRESH_BINARY_INV)

markers[background_seeds == 255] = 0

cv2.watershed(image, markers)
image[markers == -1] = [0, 0, 255]

# Muestra los objetos segmentados
cv2.imshow('Segmentation', image)

# Espera hasta que se presione la tecla 'Esc' para continuar
while True:
    key = cv2.waitKey(10)
    if key == 27:
        break

# Cierra todas las ventanas
cv2.destroyAllWindows()
