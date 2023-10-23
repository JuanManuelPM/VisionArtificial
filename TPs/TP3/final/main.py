import cv2
import numpy as np


def update_threshold(threshold_value):
    _, segmented_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imshow('Segmented Nuclei', segmented_image)

    # Encuentra y cuenta las células en la imagen segmentada
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segmented_image, connectivity=8)

    cell_count = num_labels - 1  # Restamos 1 para excluir el fondo
    cv2.putText(segmented_image, f'Cell Count: {cell_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Genera el mapa de semillas
    seed_map = np.uint8(labels / num_labels * 255)
    colored_seed_map = cv2.applyColorMap(seed_map, cv2.COLORMAP_JET)
    cv2.imshow('Colored Seed Map', colored_seed_map)
    cv2.imshow('Segmented Nuclei', segmented_image)


def update_background_threshold(threshold_value):
    _, background_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imshow('Background Seed', background_mask)


image = cv2.imread('levadura.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Segmented Nuclei', cv2.WINDOW_NORMAL)
cv2.namedWindow('Colored Seed Map', cv2.WINDOW_NORMAL)
cv2.namedWindow('Background Seed', cv2.WINDOW_NORMAL)

cv2.createTrackbar('Threshold', 'Segmented Nuclei', 0, 255, update_threshold)
cv2.createTrackbar('Background Threshold', 'Background Seed', 0, 255, update_background_threshold)

initial_threshold = 128
initial_background_threshold = 100

update_threshold(initial_threshold)
update_background_threshold(initial_background_threshold)

cv2.imshow('Original Image', image)

while True:
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
