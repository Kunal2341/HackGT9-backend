import cv2
import os 
import numpy as np
from matplotlib import pyplot as plt

imagesFolder = "ex-images"
example_img = os.path.join(imagesFolder, "1.jpg")

boundingBox = [[63, 420, 186, 214], [334, 316, 196, 190], [421, 75, 222, 171], [47, 51, 218, 214]]
img = cv2.imread(example_img)
scale_percent = 30
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dsize = (width, height)
# resize image
img = cv2.resize(img, dsize)


for shape in boundingBox:
    x, y, w, h = shape
    croppedImg = img[y:y+h, x:x+w]
    #cv2.imshow("Cropped", croppedImg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    gray = cv2.cvtColor(croppedImg, cv2.COLOR_BGR2GRAY)
    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    # list for soring names of shapes
    for contour in contours:
        # here we are ignoring first counter because 
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue
        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        # using drawContours() function
        cv2.drawContours(croppedImg, [contour], 0, (0, 0, 255), 5)
    
        # finding center point of shape
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
    
        # putting shape name at center of each shape
        if len(approx) == 3:
            cv2.putText(croppedImg, 'Triangle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
        elif len(approx) == 4:
            cv2.putText(croppedImg, 'Quadrilateral', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
        elif len(approx) == 5:
            cv2.putText(croppedImg, 'Pentagon', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
        elif len(approx) == 6:
            cv2.putText(croppedImg, 'Hexagon', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
        else:
            cv2.putText(croppedImg, 'circle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # displaying the image after drawing contours
    cv2.imshow('shapes', croppedImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()