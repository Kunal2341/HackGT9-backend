import cv2
import os 
import numpy as np
from matplotlib import pyplot as plt

def intersects(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[1] > box2[3] or box1[3] < box2[1])


LOWER_BLUE_COLOR = [25,25,0]
UPPER_BLUE_COLOR = [255,255,255]
CONTOUR_SIZE_RESTRICTION = 40
BORDER_SHAPE_PERCENT = 0.03


imagesFolder = "ex-images"
example_img = os.path.join(imagesFolder, "2.jpg")


#List of xy and width 
imgO = cv2.imread(example_img)

"""
# converting image into grayscale image
scale_percent = 30
width = int(imgO.shape[1] * scale_percent / 100)
height = int(imgO.shape[0] * scale_percent / 100)
dsize = (width, height)
# resize image
img = cv2.resize(imgO, dsize)
"""
# Get image from between 2 main colors 
imghsv = cv2.cvtColor(imgO, cv2.COLOR_BGR2HSV)
lower_blue = np.array(LOWER_BLUE_COLOR) 
upper_blue = np.array(UPPER_BLUE_COLOR)
mask_blue = cv2.inRange(imghsv, lower_blue, upper_blue)

contours, hierarchy = cv2.findContours(mask_blue, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#Show masked image
cv2.imwrite("MaskedImg.png" , mask_blue)

boundingBox = [[2333, 3975, 103, 76], [1863, 3951, 50, 90], [594, 3139, 688, 601], [1502, 2599, 609, 589], [583, 2594, 623, 603], [2061, 2495, 384, 540], [1493, 2203, 468, 514], [2022, 2003, 402, 508], [570, 1700, 600, 693], [1437, 1248, 568, 618], [2226, 969, 402, 552], [1742, 438, 563, 540], [474, 294, 765, 789]]

shapeCT = 0
for shape in boundingBox:
    x, y, w, h = shape
    croppedImg = mask_blue[y:y+h, x:x+w]
    cv2.imwrite("croppedImages/croppedImg" + str(shapeCT) + ".png", croppedImg)
    shapeCT +=1


    pixels = cv2.countNonZero(croppedImg) # OR
    image_area = croppedImg.shape[0] * croppedImg.shape[1]
    area_ratio = (pixels / image_area) * 100
    print("Saved" + "croppedImages/croppedImg" + str(shapeCT) + ".png")
    print("\t" + str(round(area_ratio, 3)))

    # We can play around with area ratio to calculate the random - that will mean how dense the drawing is

    M = cv2.moments(croppedImg)
    print("center X : '{}'".format(round(M['m10'] / M['m00'])))
    print("center Y : '{}'".format(round(M['m01'] / M['m00'])))

    # This is the center of the masked image compared to the center of the frame to measure center density

    print(math.dist([round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])], [int(croppedImg.shape[1]/2), int(croppedImg[0]/2)]))

for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approximatedShape = cv2.approxPolyDP(contour, 0.02 * perimeter, True) # Play around with this nymber
    
    if perimeter > 40:
        print(perimeter)
        print(len(approximatedShape))


"""
#Doesnt work
FloodCopy = mask_blue.copy()
mask = np.zeros((mask_blue.shape[0] + 2, mask_blue.shape[1] + 2), np.uint8)
cv2.floodFill(mask_blue, mask, (0,0), (255,255,255))
cv2.imwrite("HI.png",mask)
"""

#https://medium.com/analytics-vidhya/contours-and-convex-hull-in-opencv-python-d7503f6651bc