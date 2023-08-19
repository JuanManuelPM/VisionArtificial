import cv2
import numpy as np

# Función callback para los trackbars
def nothing(x):
    pass

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Crear una ventana para los trackbars
cv2.namedWindow('Trackbars')
cv2.createTrackbar('Threshold', 'Trackbars', 100, 255, nothing)
cv2.createTrackbar('Kernel Size', 'Trackbars', 1, 10, nothing)

while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Obtener valores actuales de los trackbars
    threshold_value = cv2.getTrackbarPos('Threshold', 'Trackbars')
    kernel_size = cv2.getTrackbarPos('Kernel Size', 'Trackbars')

    # Aplicar thresholding
    _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Aplicar operaciones morfológicas (erosión y dilatación)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    morphed = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

    # Mostrar los frames en ventanas
    cv2.imshow('Original', frame)
    cv2.imshow('Thresholded', thresholded)
    cv2.imshow('Morphed', morphed)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

