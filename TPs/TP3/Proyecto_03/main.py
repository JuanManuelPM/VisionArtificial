import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



def binary(val):

    #Punto 1
    img = cv.imread('TPs/Proyecto_03/images/levadura2.png')
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, val, 255, cv.THRESH_BINARY)
    bin_img = cv.morphologyEx(thresh, 
                           cv.MORPH_OPEN,
                           kernel,
                           iterations=2)
    cv.imshow("img", thresh)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cant_contours= len(contours)
    

    cv.putText(img, f"Cantidad: {cant_contours}", (200,200), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    #Eliminacion del ruido
    
    opening = cv.morphologyEx(bin_img, cv.MORPH_OPEN, kernel, iterations=2)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=2)
    sure_bg = cv.dilate(closing, kernel, iterations=3)
    sure_fg = cv.erode(closing, kernel, iterations=3)

    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    # sure background area
    sure_bg = cv.dilate(bin_img, kernel, iterations=3)
    
      
    # Distance transform
    dist = cv.distanceTransform(bin_img, cv.DIST_L2, 5)
    
      
    #foreground area
    ret, sure_fg = cv.threshold(dist, 0.5 * dist.max(), 255, cv.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)  
    
      
    # unknown area
    unknown = cv.subtract(sure_bg, sure_fg)
    

    
    # Marker labelling
    _, markers = cv.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers += 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0


    markers = cv.watershed(img, markers)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(markers, cmap="tab20b")
    ax.axis('off')
    plt.show()
    img[markers == -1] = [255, 0, 0]

    labels = np.unique(markers)
  
    cells = []
    for label in labels[2:]:  
  
      # Create a binary image in which only the area of the label is in the foreground 
      #and the rest of the image is in the background   
      target = np.where(markers == label, 255, 0).astype(np.uint8)
    
      # Perform contour extraction on the created binary image
      contours, hierarchy = cv.findContours(
        target, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
      )
      cells.append(contours[0])
  
    # Draw the outline
    img = cv.drawContours(img, cells, -1, color=(0, 23, 223), thickness=2)





    colormap = cv.applyColorMap(img, cv.COLORMAP_PLASMA)
    '''
    mask = np.zeros(img.shape[:2], np.uint8)
    rect = (50,50,450,290)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 10)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    plt.imshow(img)
    '''
    markers = cv.watershed(img, markers)
    

      # Puedes cambiar COLORMAP_VIRIDIS por el mapa de colores que prefieras.
    cv.imshow("Colormap", colormap)

    cv.waitKey()



cv.namedWindow('img')
cv.createTrackbar('Trackbar', 'img', 0, 100, binary)
cv.setTrackbarPos('Trackbar', 'img', 80)

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