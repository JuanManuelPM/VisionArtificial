import cv2
import math
import numpy as np
from joblib import load
from label_conv import int_to_label
from training_model import train_model

# carga el modelo
clasificador = load('filename.joblib')


val = 0
model = train_model()


def get_contour(image, name):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Any sharp edges in images are smoothed while minimizing too much blurring.
    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # We also used THRESH_OTSU to analyze the image and determine the threshold.
    ret3, thresh = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contour, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contour, -1, (0, 255, 0), 2)
    cv2.imshow("image" + name, image)
    return contour


# Callback function for the kernel size trackbar
def update_kernel_size(value):
    global kernel_size
    if value == 0:
        kernel_size = 1
    kernel_size = value


def generateHu(listaHu):
    toR = []
    for i in listaHu:
        toR.append(i[0])

    return toR


def compare(con1):
    moments = cv2.moments(con1)
    huMoments = cv2.HuMoments(moments)

    # Log scale hu moments
    for i in range(0, 7):
        huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(
            abs(huMoments[i]))  # Mapeo para agrandar la escala.
    hu_moments = huMoments

    sample = np.array([hu_moments], dtype=np.float32)  # numpy
    #predictedLabel
    testResponse = model.predict(sample)[1]  # Predice la clase de cada file
    return str(testResponse)


# Callback function for the binary threshold trackbar
def update_threshold(value):
    global threshold_value
    threshold_value = value


# Initialize camera capture
cap = cv2.VideoCapture(0)  # 0 represents the default camera

# Create a window for displaying the camera feed
cv2.namedWindow('Camera')

# Create trackbars for kernel size, binary threshold, and contour detection
cv2.createTrackbar('Kernel Size', 'Camera', 1, 20, update_kernel_size)
cv2.createTrackbar('Threshold', 'Camera', 128, 255, update_threshold)

codigo = {"1": "Star", "2": "Rectangle", "3": "Triangle"}

kernel_size = 1  # Initial kernel size
threshold_value = 128  # Initial threshold value

while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        print("Error reading frame")
        break

    # Convert to binary frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create a kernel for morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Apply binary thresholding to the grayscale frame
    _, binary_frame = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Apply opening operation to denoise the image
    fixed_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel)

    # Find and filter contours if contour_detection is enabled
    contours, hierarchy = cv2.findContours(fixed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    valid_contours_triangles = []
    valid_contours_rectangles = []
    valid_contours_stars = []
    valid_contours_lightning = []
    invalid_contours = []

    for contour in contours:
        # Calculate the contour's area to filter out the bigger ones
        area = cv2.contourArea(contour)
        if 100 < area < 5000:
            comparacion = compare(contour)
            if comparacion[1] == '1':
                valid_contours_rectangles.append(contour)  # rectangle detected
            elif comparacion[1] == '2':
                valid_contours_triangles.append(contour)  # triangle detected
            elif comparacion[1] == '3':
                valid_contours_stars.append(contour)  # stars detected
            elif comparacion[1] == '4':
                valid_contours_lightning.append(contour)  # lightning detected
            else:
                invalid_contours.append(contour)

    # Add labels to the detected shapes
    for contour in valid_contours_stars:
        x, y = contour[0][0]
        cv2.putText(frame, "Star", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    for contour in valid_contours_rectangles:
        x, y, _, _ = cv2.boundingRect(contour)
        cv2.putText(frame, "Square", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    for contour in valid_contours_triangles:
        x, y = contour[0][0]
        cv2.putText(frame, "Triangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for contour in valid_contours_lightning:
        x, y = contour[0][0]
        cv2.putText(frame, "Lightning", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Add labels to the detected but unrecognized shapes
    for contour in invalid_contours:
        x, y = contour[0][0]
        cv2.putText(frame, "?", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw contours in fixed frame
    cv2.drawContours(frame, valid_contours_stars, -1, (0, 255, 0), 2)
    cv2.drawContours(frame, valid_contours_triangles, -1, (255, 0, 0), 2)
    cv2.drawContours(frame, valid_contours_rectangles, -1, (0, 255, 255), 2)
    cv2.drawContours(frame, valid_contours_lightning, -1, (10, 50, 255), 2)
    cv2.drawContours(frame, invalid_contours, -1, (0, 0, 255), 2)

    # Convert binary to 3-frame channel to show
    binary_frame_3channel = cv2.cvtColor(fixed_frame, cv2.COLOR_GRAY2BGR)

    # Combine both binary and original frame to show
    combined_frame = np.hstack((binary_frame_3channel, frame))

    # Display the combined frame
    cv2.imshow('Camera', combined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to exit
        break

# Release the camera capture resources and close the window
cap.release()
cv2.destroyAllWindows()
