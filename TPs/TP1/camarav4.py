import cv2
import numpy as np

image_circle = cv2.imread('../../static/images/circulo.jpg')
image_triangle = cv2.imread('../../static/images/triangulo.jpeg')
image_square = cv2.imread('../../static/images/cuadrado.jpeg')



gray = cv2.cvtColor(image_circle, cv2.COLOR_BGR2GRAY)
# Any sharp edges in images are smoothed while minimizing too much blurring.
blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
# We also used THRESH_OTSU to analyze the image and determine the threshold.
ret3, thresh = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contour_circle, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image_circle, contour_circle, -1, (0, 255, 0), 2)
cv2.imwrite('image_circle.jpg', image_circle)

# Callback function for the kernel size trackbar
def update_kernel_size(value):
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
    _, binary_frame = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY)

    # Apply opening operation to denoise the image
    denoised_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel)

    # Find and filter contours if contour_detection is enabled
    contours, hierarchy = cv2.findContours(denoised_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area:
            print(cv2.matchShapes(contour, contour_circle[0], cv2.CONTOURS_MATCH_I2, 0))
            if cv2.matchShapes(contour, contour_circle[0], cv2.CONTOURS_MATCH_I2, 0) < 0.01:
                valid_contours.append(contour)

    # Draw contours in denoised frame
    cv2.drawContours(frame, valid_contours, -1, (0, 0, 255), 2)

    # Convert binary to 3-frame channel to show
    binary_frame_3channel = cv2.cvtColor(denoised_frame, cv2.COLOR_GRAY2BGR)

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
