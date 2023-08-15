import cv2

import cv2 as cv


def update_brightness(value):
    global brightness
    brightness = value


def update_kernel_size(value):
    global kernel
    kernel = value

def update_gray_value(value):
    global val
    val = value

cap = cv2.VideoCapture(0)  # 0 represents the default camera

cv.namedWindow('Camera')

cv.createTrackbar('Brightness', 'Camera', 0, 100, update_brightness)

cv2.createTrackbar('Kernel Size', 'Camera', 1, 20, update_kernel_size)

cv2.createTrackbar('Binary Threshold', 'Camera', 0, 255, update_gray_value)

brightness = 0  # Initial brightness value

kernel_size = 1  # Initial kernel size

val = 1  # Initial binary-gray value

while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        print("Error reading frame")
        break

    # Convert the frame to grayscale for denoising
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret2, thresh2 = cv.threshold(gray_frame, val, 255, cv.THRESH_BINARY_INV)

    # Create a kernel for morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Apply opening operation to denoise the image
    denoise_frame = cv2.morphologyEx(gray_frame, cv2.MORPH_OPEN, kernel)

    closing = cv.morphologyEx(denoise_frame, cv.MORPH_CLOSE, kernel)

    adjusted_frame = cv2.add(frame, val)

    cv.imshow('Camera', adjusted_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv.destroyAllWindows()