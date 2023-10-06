import numpy as np
import cv2 as cv



def binary(val):

    #Punto 1
    img = cv.imread('TPs/Proyecto_03/images/levadura2.png')

    #Se convierte la imagen en 
    #color (BGR, que es el formato de color comúnmente utilizado por OpenCV)
    #en una imagen en escala de grises.

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #se aplica un umbral a la imagen en escala de grises 'gray' para
    #convertirla en una imagen binaria
    _, thresh = cv.threshold(gray, val, 255, cv.THRESH_BINARY)
    cv.imshow("img", thresh)

    #Esta línea de código se utiliza para encontrar y calcular los contornos en una imagen binaria 
    #'thresh' y almacenar esos contornos en la variable 'contours'. 
    #Además, también se almacena información sobre la jerarquía
    #de los contornos en la variable 'hierarchy'.
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cant_contours= len(contours)


    


    #Eliminacion del ruido

    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=2)

    sure_bg = cv.dilate(closing, kernel, iterations=3)
    sure_fg = cv.erode(closing, kernel, iterations=3)



    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    cv.putText(img, f"Cantidad: {cant_contours}", (100,700), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Marker labelling
    _, markers = cv.connectedComponents(sure_bg)
    print(markers)
    print("------------------")
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(img, markers)

    print(img)
    print(markers)
    result_img = img.copy()

    # Asignar color a las regiones etiquetadas como -1 en la copia de la imagen
    result_img[markers == 1] = [0, 0, 255]
    result_img[markers == 0] = [0, 255, 0]  # Color para las regiones segmentadas en rojo

    # Aplicar el mapa de colores a la imagen en escala de grises resultante
    colormap = cv.applyColorMap(gray, cv.COLORMAP_TWILIGHT_SHIFTED)  # Puedes elegir un mapa de colores diferente
    cv.imshow("Colormap", colormap)

    # Mostrar la imagen resultante con regiones segmentadas coloreadas
    cv.imshow("Segmented Image", result_img)

    cv.waitKey()



cv.namedWindow('img')
cv.createTrackbar('Trackbar', 'img', 0, 100, binary)
binary(0)

# noise removal


# sure background area

# Finding sure foreground area


# SE DEBERIA USAR EL DISTANCE TRANSFORM ANTES Q EL ERODE
# PQ LOS OBJETOS ESTAN TODOS PEGADOS PERO DE BAJA PQ NOSE SI HAY Q EXPLCIAR ESO

# dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
# ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

while True:
    key = cv.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # Salir con 'q' o 'Esc'
        break

# Cerrar todas las ventanas
cv.destroyAllWindows()