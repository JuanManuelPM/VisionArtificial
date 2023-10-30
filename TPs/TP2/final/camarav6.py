import cv2
import math
import numpy as np
from joblib import load

from trainer import get_trained_model_alt, get_trained_model, read_labels
from hu_moments_generation import generate_hu_moments_file

val = 0
generate_hu_moments_file()
model = get_trained_model_alt()
initial_min_area = 10
initial_max_area = 301
min_area = initial_min_area
max_area = initial_max_area


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

    sample = np.array([(cv2.HuMoments(moments))], dtype=np.float32)
    sample = sample.flatten()
    for i in range(0, 7):
        if sample[i] > 0 or sample[i] < 0:
            sample[i] = -1 * math.copysign(1, sample[i]) * math.log10(abs(sample[i]))
        else:
            sample[i] = 0
    testResponse = model.predict(sample.reshape(1, -1))[0]
    return int(testResponse)


# Callback function for the binary threshold trackbar
def update_threshold(value):
    global threshold_value
    threshold_value = value
def update_Min_area(value):
    global min_area
    min_area = value
def update_Max_area(value):
    global max_area
    max_area = value

# Initialize camera capture
cap = cv2.VideoCapture(0)  # 0 represents the default camera

# Create a window for displaying the camera feed
cv2.namedWindow('Camera')

# Create trackbars for kernel size, binary threshold, and contour detection
cv2.createTrackbar('Kernel Size', 'Camera', 1, 20, update_kernel_size)
cv2.createTrackbar('Threshold', 'Camera', 128, 255, update_threshold)
cv2.createTrackbar('Min_Area', 'Camera', 0, 300, update_Min_area)
cv2.createTrackbar('Max_Area', 'Camera', 301, 100000, update_Max_area)

codigo = {"1": "Star", "2": "Square", "3": "Triangle", "4": "Invalid"}

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
    invalid_contours = []

    for contour in contours:
        # Calculate the contour's area to filter out the bigger ones
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            comparacion = compare(contour)
            if comparacion == 2:
                valid_contours_rectangles.append(contour)  # rectangle detected
            elif comparacion == 3:
                valid_contours_triangles.append(contour)  # triangle detected
            elif comparacion == 1:
                valid_contours_stars.append(contour)  # stars detected
            else:
                invalid_contours.append(contour)

    # Add labels to the detected shapes
    for contour in valid_contours_stars:
        x, y = contour[0][0]
        cv2.putText(frame, "Star", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for contour in valid_contours_rectangles:
        x, y, _, _ = cv2.boundingRect(contour)
        cv2.putText(frame, "Square", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for contour in valid_contours_triangles:
        x, y = contour[0][0]
        cv2.putText(frame, "Triangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    # Add labels to the detected but unrecognized shapes
    for contour in invalid_contours:
        x, y = contour[0][0]
        cv2.putText(frame, "?", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Draw contours in fixed frame
    cv2.drawContours(frame, valid_contours_stars, -1, (0, 255, 0), 2)
    cv2.drawContours(frame, valid_contours_triangles, -1, (255, 0, 0), 2)
    cv2.drawContours(frame, valid_contours_rectangles, -1, (0, 0, 255), 2)
    cv2.drawContours(frame, invalid_contours, -1, (0, 255, 255), 2)

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
