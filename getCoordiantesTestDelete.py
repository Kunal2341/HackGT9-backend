import cv2
import numpy as np
import os
import shapely
def intersects(box1, box2):
    """Helper function for get_coordinates"""

    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[1] > box2[3] or box1[3] < box2[1])



path_to_image = "ex-images/canvasImage.jpeg"

LOWER_BLUE_COLOR = [25,25,0]
UPPER_BLUE_COLOR = [255,255,255]
CONTOUR_SIZE_RESTRICTION = 40
BORDER_SHAPE_PERCENT = 0.05
imgO = cv2.imread(path_to_image)

# converting image into grayscale image
scale_percent = 30
width = int(imgO.shape[1] * scale_percent / 100)
height = int(imgO.shape[0] * scale_percent / 100)
dsize = (width, height)
img = cv2.resize(imgO, dsize)

# Get image from between 2 main colors 
imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array(LOWER_BLUE_COLOR) 
upper_blue = np.array(UPPER_BLUE_COLOR)
mask_blue = cv2.inRange(imghsv, lower_blue, upper_blue)
contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

im = np.copy(img)
dimensionsShapes = []

for c in contours:
    if (cv2.contourArea(c) > CONTOUR_SIZE_RESTRICTION): 
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        dimensionsShapes.append([x,y,w,h])

# Checks each value with all others in arrary to see intersection, removes smaller array
for shape in dimensionsShapes:
    for shapeCheck in dimensionsShapes:
        if intersects((shape[0],shape[1],shape[2]+shape[0],shape[3]+shape[1]), (shapeCheck[0],shapeCheck[1],shapeCheck[2]+shapeCheck[0],shapeCheck[3]+shapeCheck[1])) and shape != shapeCheck:
            if shape[2]*shape[3] > shapeCheck[2]*shapeCheck[3]:
                dimensionsShapes.remove(shapeCheck)
            else:
                dimensionsShapes.remove(shape)

bufferedDimensions = []
savingImg = imgO
for shapeFinalizaed in dimensionsShapes:
    border = max(int(w * imgO.shape[1] / img.shape[1] * BORDER_SHAPE_PERCENT), 
                int(h * imgO.shape[0] / img.shape[0] * BORDER_SHAPE_PERCENT))
    x, y, w, h = shapeFinalizaed 
    x = max(int(x * imgO.shape[1] / img.shape[1] - border),0)
    y = max(int(y * imgO.shape[0] / img.shape[0] - border),0)
    w = int(w * imgO.shape[1] / img.shape[1]) + border * 2
    h = int(h * imgO.shape[0] / img.shape[0]) + border * 2

    bufferedDimensions.append([x,y,w,h])
print(bufferedDimensions)
