from base64 import encode
import os, cv2

img_file = os.listdir(".\\imagenes")

hus = {}

def get_biggest_contour(contours):
    try:
        max_cnt = contours[0]
        for cnt in contours:
            if cv2.contourArea(cnt) > cv2.contourArea(max_cnt):
                max_cnt = cnt
        return max_cnt
    except:
        return contours

for f in img_file:
    img = cv2.imread(".\\imagenes\\"+f)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret1, img = cv2.threshold(img,60, 255, cv2.THRESH_BINARY_INV)
    imgContorno, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    hus[f.split(".")[0]] = cv2.HuMoments(cv2.moments(get_biggest_contour(imgContorno)))

with open("HuMom.txt", "w", encoding="UTF-8-SIG") as f:
    for k in hus.keys():
        print(k)
        f.write("[")
        for h in hus[k]:
            f.write(f"{h[0]},")
        f.write("]\n")
