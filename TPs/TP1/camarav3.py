import cv2
import numpy as np


def update_contour(value):
    global contour_mode
    contour_mode = value


# Callback function for the kernel size trackbar
def update_kernel_size(value):
    global kernel_size
    kernel_size = value


# Callback function for the binary threshold trackbar
def update_threshold(value):
    global threshold_value
    threshold_value = value


# Initialize camera capture
cap = cv2.VideoCapture(0)  # 0 represents the default camera

# Create a window for displaying the camera feed
cv2.namedWindow('Camera')

# Create trackbars for kernel size and binary threshold
cv2.createTrackbar('Kernel Size', 'Camera', 1, 20, update_kernel_size)
cv2.createTrackbar('Threshold', 'Camera', 128, 255, update_threshold)
cv2.createTrackbar('Contour', 'Camera', 100, 500, update_contour)

kernel_size = 1  # Initial kernel size
threshold_value = 128  # Initial threshold value
contour_mode = 0  # Initial contour detection mode (off)

while True:
    ret, frame = cap.read()  # Read a frame from the camera
    frame = cv2.flip(frame, 1)

    if not ret:
        print("Error reading frame")
        break

    # Convert the frame to grayscale for denoising
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create a kernel for morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Apply opening operation to denoise the image
    denoised_frame = cv2.morphologyEx(gray_frame, cv2.MORPH_OPEN, kernel)

    # Apply binary thresholding to the denoised frame
    _, binary_frame = cv2.threshold(denoised_frame, threshold_value, 255, cv2.THRESH_BINARY)

    # Apply contour detection if enabled
    # Find contours in the binary thresholded frame
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original frame
    contour_image = frame.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), contour_threshold)

    # Combine the modified original frame, denoised frame, and contour image
    combined_frame = np.vstack((frame, denoised_frame, contour_image))

    # Display the combined frame
    cv2.imshow('Camera', combined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to exit
        break

# Release the camera capture resources and close the window
cap.release()
cv2.destroyAllWindows()
