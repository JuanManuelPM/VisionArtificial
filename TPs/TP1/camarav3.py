import cv2
import numpy as np


# Callback function for the trackbar
def update_kernel_size(value):
    global kernel_size
    kernel_size = value


# Initialize camera capture
cap = cv2.VideoCapture(0)  # 0 represents the default camera

# Create a window for displaying the camera feed
cv2.namedWindow('Camera')

# Create a trackbar named 'Kernel Size' in the 'Camera' window
cv2.createTrackbar('Kernel Size', 'Camera', 1, 20, update_kernel_size)

kernel_size = 1  # Initial kernel size

while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        print("Error reading frame")
        break

    # Convert the frame to grayscale for denoising
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create a kernel for morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Apply opening operation to denoise the image
    denoised_frame = cv2.morphologyEx(gray_frame, cv2.MORPH_OPEN, kernel)

    # Display the original and denoised frames
    cv2.imshow('Camera', np.hstack((gray_frame, denoised_frame)))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to exit
        break

# Release the camera capture resources and close the window
cap.release()
cv2.destroyAllWindows()