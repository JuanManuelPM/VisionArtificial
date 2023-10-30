import cv2
import gray
import numpy as np

# Variables globales para gestionar las semillas
seeds = np.zeros_like(gray, dtype=np.int32)
seed_count = 0

# Carga de la imagen de células
image = cv2.imread('../../static/images/celulas.png')

# Convierte la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
import numpy as np

# Función para agregar una semilla en el fondo
def add_background_seed(background_threshold):
    global seeds, seed_count
    _, thresholded = cv2.threshold(gray, background_threshold, 255, cv2.THRESH_BINARY)
    seeds[thresholded == 255] = seed_count + 1
    seed_count += 1


# Función para actualizar el mapa de semillas y visualizarlo
def update_seeds(threshold_value):
    global seeds, seed_count
    _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Encuentra y cuenta las células segmentadas
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cell_count = len(contours)

    # Inicializa el mapa de semillas
    seeds = np.zeros_like(gray, dtype=np.int32)
    seed_count = 0

    # Agrega una semilla para el fondo
    add_background_seed(128)  # Umbral de fondo inicial

    # Asigna semillas a las células segmentadas
    for i, contour in enumerate(contours):
        cv2.drawContours(seeds, [contour], -1, (i + 1), -1)

    # Aplica watershed
    markers = cv2.watershed(image, seeds)

    # Dibuja los contornos de las células en la imagen original
    result_image = image.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

    # Visualiza el mapa de semillas con colores
    seeds_colored = cv2.applyColorMap(np.uint8(255 * seeds / np.max(seeds)), cv2.COLORMAP_JET)
    seeds_colored[markers == -1] = [0, 0, 255]  # Fronteras en rojo

    # Muestra la cantidad de células encontradas en la imagen
    cv2.putText(result_image, f'Cells: {cell_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Muestra la imagen resultante con los contornos y el mapa de semillas
    cv2.imshow('Segmented Cells', np.hstack((result_image, seeds_colored)))


# Crea una ventana con trackbars para ajustar el umbral de células y el umbral de fondo
cv2.namedWindow('Segmented Cells')
cv2.createTrackbar('Cell Threshold', 'Segmented Cells', 128, 255, update_seeds)
cv2.createTrackbar('Background Threshold', 'Segmented Cells', 128, 255, add_background_seed)

# Llama a la función de umbralización y segmentación inicial
update_seeds(128)

# Espera hasta que se presione la tecla 'Esc' para salir
while True:
    key = cv2.waitKey(10)
    if key == 27:  # 27 es el código ASCII para la tecla 'Esc'
        break

# Cierra todas las ventanas
cv2.destroyAllWindows()
