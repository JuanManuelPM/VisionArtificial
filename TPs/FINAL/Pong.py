import cv2
from HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Importing all images
imgBackground = cv2.imread("resources/Background_new.png")
imgGameOver = cv2.imread("resources/gameOver.png")
imgBall = cv2.imread("resources/Ball.png")
imgBat1 = cv2.imread("resources/Sprite-0001.png", cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("resources/Sprite-0002.png", cv2.IMREAD_UNCHANGED)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Variables
ballPos = [100, 100]
speedX = 15
speedY = 15
speed_increment = 1.1  # Increment factor for speed after each bounce on players' bats
gameOver = False
score = [0, 0]

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    imgRaw = img.copy()

    # Find the hand and its landmarks
    hands, img = detector.findHands(img, flipType=False)  # with draw

    # Overlaying the background image
    img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)

    # Draw the bat overlays
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = imgBat1.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)

            if hand['type'] == "Left":
                img[y1:y1 + h1, 59:59 + w1] = imgBat1[:, :, :3]  # Overlay bat1

                if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX * speed_increment
                    ballPos[0] += 30
                    score[0] += 1

            # Draw the bat overlays
            if hands:
                for hand in hands:
                    x, y, w, h = hand['bbox']
                    h1, w1, _ = imgBat1.shape
                    y1 = y - h1 // 2
                    y1 = np.clip(y1, 20, 415)

                    if hand['type'] == "Left":
                        img[y1:y1 + h1, 59:59 + w1] = imgBat1[:, :, :3]  # Overlay bat1

                        if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                            speedX = -speedX * speed_increment
                            ballPos[0] += 30
                            score[0] += 1

                    if hand['type'] == "Right":
                        x2 = 1195 - w1 + 25  # Adjust this value to move the second bat to the right
                        img[y1:y1 + h1, x2:x2 + w1] = imgBat2[:, :, :3]  # Overlay bat2

                        if x2 < ballPos[0] < x2 + w1 and y1 < ballPos[1] < y1 + h1:
                            speedX = -speedX * speed_increment
                            ballPos[0] -= 30
                            score[1] += 1

    # Game Over
    if ballPos[0] < 40 or ballPos[0] > 1200:
        gameOver = True

    if gameOver:
        img = imgGameOver
        cv2.putText(img, str(score[1] + score[0]).zfill(2), (585, 360), cv2.FONT_HERSHEY_COMPLEX,
                    2.5, (200, 0, 200), 5)

    # If game not over move the ball
    else:
        # Move the Ball
        if ballPos[1] >= 500 or ballPos[1] <= 10:
            speedY = -speedY

        ballPos[0] += speedX
        ballPos[1] += speedY

        # Draw the ball
        cv2.circle(img, (int(ballPos[0]), int(ballPos[1])), 20, (0, 0, 0), -1)

        cv2.putText(img, str(score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(img, str(score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)

    img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        ballPos = [100, 100]
        speedX = 15
        speedY = 15
        speed_increment = 1.1  # Reset speed increment
        gameOver = False
        score = [0, 0]
        imgGameOver = cv2.imread("resources/gameOver.png")
