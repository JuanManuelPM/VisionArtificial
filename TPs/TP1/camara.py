import cv2

import cv2 as cv

def update_brightness(value):
    global brightness
    brightness = value
def denoise(frame, method, radius):
    kernel = cv.getStructuringElement(method, (radius, radius))
    opening = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    return closing

def getFrame():
    return cap.read()

cap = cv2.VideoCapture(0)  # 0 represents the default camera

cv.namedWindow('Camera')

cv.createTrackbar('Brightness', 'Camera', 0, 100, update_brightness)
cv.createTrackbar('Brightness', 'Camera', 0, 100, denoise(getFrame(), cv.MORPH_ELLIPSE, 1))


brightness = 0  # Initial brightness value

while True:
    ret, frame = cap.read()  # Read a frame from the camera

    adjusted_frame = cv2.add(frame, brightness)

    cv.imshow('Camera', adjusted_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv.destroyAllWindows()





