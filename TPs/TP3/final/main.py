import cv2 as cv
import numpy as np

# VENTANAS
window_name = 'threshold'
watershed_window_name = 'watershed'

# KERNEL
kernel = np.ones((3, 3), np.uint8)

# TRACKBARS
trackbar_window_name = 'Trackbars'
cv.namedWindow(trackbar_window_name)

max_value_thresh = 255
thresh_name = 'Threshold'
cv.createTrackbar(thresh_name, trackbar_window_name, 0, max_value_thresh, lambda x: None)

max_value_morph = 15
dilate_name = 'Dilate'
erode_name = 'Erode'
cv.createTrackbar(erode_name, trackbar_window_name, 0, max_value_morph, lambda x: None)
cv.createTrackbar(dilate_name, trackbar_window_name, 0, max_value_morph, lambda x: None)

# Area classification threshold
#area_threshold = 200

while True:
    # Image
    img = cv.imread('levadura.png')
    img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Segmentation with threshold
    thresh_value = cv.getTrackbarPos(thresh_name, trackbar_window_name)

    _, thresh = cv.threshold(img2, thresh_value, max_value_thresh, cv.THRESH_BINARY)

    # Noise removal
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=3)
    closing = cv.morphologyEx(opening, cv.MORPH_OPEN, kernel, iterations=3)

    cv.imshow('threshold', thresh)

    # Seeds
    erode_value = cv.getTrackbarPos(erode_name, trackbar_window_name)
    dilate_value = cv.getTrackbarPos(dilate_name, trackbar_window_name)

    # Finding sure foreground area sureFG
    foreground = cv.erode(closing, kernel, iterations=erode_value)
    background = cv.dilate(closing, kernel, iterations=dilate_value)
    # Finding unknown region
    unknown = cv.subtract(background, foreground)

    # Marker labeling from the foreground area surface
    number, markers = cv.connectedComponents(foreground)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of the unknown with zero
    markers[unknown == 255] = 0

    map = np.uint8(255 * markers / number)
    color_map = cv.applyColorMap(map, cv.COLORMAP_JET)
    cv.imshow('color_map', color_map)

    # Segmentation with watershed
    markers = cv.watershed(img, markers)
    img[markers == -1] = [0, 0, 255]

    # Count cells
    numberSTR = str(number - 1)
    number_cell = 'Cell count: ' + numberSTR
    cv.putText(img, number_cell, (10, 800), cv.FONT_HERSHEY_SIMPLEX, 1, color=(0, 0, 255))

    # Create a result_image for area classification
    result_image = img.copy()

    # Area classification
    contours, hierachy = cv.findContours(image=foreground, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    cell_count = 0
    for contour in contours:
        #if cv.contourArea(contour) < area_threshold:
           # continue
        cell_count += 1
        cv.drawContours(result_image, [contour], -1, (0, 255, 0), 2)
        # Calculate and display the area of each cell
        M = cv.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv.putText(result_image, f'Area: {cv.contourArea(contour)}', (cX - 20, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1)
    cv.imshow('Area classification', result_image)

    cv.imshow(watershed_window_name, img)

    tecla = cv.waitKey(30)
    if tecla == 27:
        break
