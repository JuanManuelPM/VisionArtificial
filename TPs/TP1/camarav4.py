import cv2
import numpy as np


# Callback function for the kernel size trackbar
def update_kernel_size(value):
    if(value == 0):
        value = 1
    global kernel_size
    kernel_size = value


# Callback function for the binary threshold trackbar
def update_threshold(value):
    global threshold_value
    threshold_value = value


# Callback function for the contour detection trackbar
def update_contour_detection(value):
    global contour_detection
    contour_detection = value


# Initialize camera capture
cap = cv2.VideoCapture(0)  # 0 represents the default camera

# Create a window for displaying the camera feed
cv2.namedWindow('Camera')

# Create trackbars for kernel size, binary threshold, and contour detection
cv2.createTrackbar('Kernel Size', 'Camera', 1, 20, update_kernel_size)
cv2.createTrackbar('Threshold', 'Camera', 128, 255, update_threshold)
cv2.createTrackbar('Contour Detection', 'Camera', 0, 1, update_contour_detection)

kernel_size = 1  # Initial kernel size
threshold_value = 128  # Initial threshold value
contour_detection = 0  # Initial contour detection value

while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        print("Error reading frame")
        break

    # Create a kernel for morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Apply opening operation to denoise the image
    denoised_frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)

    # Convert the denoised frame to grayscale for binary thresholding
    gray_denoised_frame = cv2.cvtColor(denoised_frame, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding to the grayscale denoised frame
    _, binary_frame = cv2.threshold(gray_denoised_frame, threshold_value, 255, cv2.THRESH_BINARY)

    # Convert the binary frame to a 3-channel (color) image
    colored_binary_frame = cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2BGR)

    # Find and draw contours if contour_detection is enabled
    if contour_detection == 1:
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Combine the original, denoised, binary thresholded, and contour frames horizontally
    combined_frame = np.hstack((frame, denoised_frame, colored_binary_frame))

    # Display the combined frame
    cv2.imshow('Camera', combined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to exit
        break

# Release the camera capture resources and close the window
cap.release()
cv2.destroyAllWindows()
