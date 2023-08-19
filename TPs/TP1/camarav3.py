import cv2
import numpy as np


# Callback function for the kernel size trackbar
def update_kernel_size(value):
    global kernel_size
    kernel_size = value


# Callback function for the binary threshold trackbar
def update_threshold(value):
    global threshold_value
    threshold_value = value


# Callback function for the contour trackbar
def update_contour(value):
    global contour_threshold
    contour_threshold = value


# Initialize camera capture
cap = cv2.VideoCapture(0)  # 0 represents the default camera

# Create a window for displaying the camera feed
cv2.namedWindow('Camera')

# Create trackbars for kernel size, binary threshold, and contour threshold
cv2.createTrackbar('Kernel Size', 'Camera', 1, 20, update_kernel_size)
cv2.createTrackbar('Threshold', 'Camera', 128, 255, update_threshold)
cv2.createTrackbar('Contour Threshold', 'Camera', 100, 500, update_contour)

kernel_size = 1  # Initial kernel size
threshold_value = 128  # Initial threshold value
contour_threshold = 100  # Initial contour threshold value

while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        print("Error reading frame")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the grayscale frame to match the dimensions of the color frame
    gray_frame_resized = cv2.resize(gray_frame, (frame.shape[1], frame.shape[0]))

    # Apply denoising using kernel size
    denoised_frame = cv2.GaussianBlur(gray_frame_resized, (kernel_size, kernel_size), 0)

    # Apply binary thresholding
    _, binary_frame = cv2.threshold(denoised_frame, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours and draw them on the original frame
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = frame.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), contour_threshold)

    # Combine the images horizontally using cv2.hconcat
    combined_frame = cv2.hconcat([frame, cv2.cvtColor(denoised_frame, cv2.COLOR_GRAY2BGR), contour_image])

    # Display the combined frame
    cv2.imshow('Camera', combined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to exit
        break

# Release the camera capture resources and close the window
cap.release()
cv2.destroyAllWindows()
