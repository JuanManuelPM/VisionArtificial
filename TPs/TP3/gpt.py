import cv2
import numpy as np

# Carga de la imagen de células
image = cv2.imread('../../static/images/celulas.png')


# Convierte la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define las variables para los umbrales de fondo y área
background_threshold = 100  # Umbral de fondo (puedes cambiarlo)
area_threshold = 2000  # Umbral de área (puedes cambiarlo)


# Define la función de umbralización para fondo y área
def update_background_threshold(value):
    global background_threshold
    background_threshold = value
    perform_segmentation()


def update_area_threshold(value):
    global area_threshold
    area_threshold = value
    perform_segmentation()


# Define la función de segmentación completa
def perform_segmentation():
    _, thresholded = cv2.threshold(gray, background_threshold, 255, cv2.THRESH_BINARY)

    # Encuentra y cuenta las células segmentadas
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cell_count = 0
    result_image = image.copy()

    for contour in contours:
        if cv2.contourArea(contour) < area_threshold:
            continue
        cell_count += 1
        cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)
        # Calcula y muestra el área de cada célula
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(result_image, f'Area: {cv2.contourArea(contour)}', (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)

    # Muestra la cantidad de células encontradas en la imagen
    cv2.putText(result_image, f'Cells: {cell_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Muestra la imagen resultante
    cv2.imshow('Segmented Cells', result_image)


# Crea ventanas con trackbars para ajustar los umbrales de fondo y área
cv2.namedWindow('Segmented Cells')
cv2.createTrackbar('Background Threshold', 'Segmented Cells', background_threshold, 255, update_background_threshold)
cv2.createTrackbar('Area Threshold', 'Segmented Cells', area_threshold, 5000, update_area_threshold)

# Inicializa la ventana con valores de umbral por defecto
cv2.setTrackbarPos('Background Threshold', 'Segmented Cells', background_threshold)
cv2.setTrackbarPos('Area Threshold', 'Segmented Cells', area_threshold)

# Realiza la segmentación inicial
perform_segmentation()

# Espera hasta que se presione la tecla 'Esc' para salir
while True:
    key = cv2.waitKey(10)
    if key == 27:  # 27 es el código ASCII para la tecla 'Esc'
        break

# Cierra todas las ventanas
cv2.destroyAllWindows()
