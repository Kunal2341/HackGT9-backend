import cv2
import os 
import numpy as np
from matplotlib import pyplot as plt

imagesFolder = "ex-images"
example_img = os.path.join(imagesFolder, "1.jpg")


#List of xy and width 

img = cv2.imread(example_img)

# converting image into grayscale image
scale_percent = 30
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dsize = (width, height)
# resize image
img = cv2.resize(img, dsize)

def intersects(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[1] > box2[3] or box1[3] < box2[1])

# Get image from between 2 main colors 
imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([25,25,0]) 
upper_blue = np.array([255,255,255])
mask_blue = cv2.inRange(imghsv, lower_blue, upper_blue)
#Show masked image
# cv2.imshow("winname" , mask_blue)
contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

imKnown = np.copy(img)

x, y, w, h = 220, 441, 26, 29
cv2.rectangle(imKnown,(x,y),(x+w,y+h),(255,0,0),2)
cv2.putText(imKnown,str("orginal"),(x+w+10,y+h),0,0.3,(255,0,0))

x, y, w, h = 83, 440, 146, 174
cv2.rectangle(imKnown,(x,y),(x+w,y+h),(0,255,0),2)
cv2.putText(imKnown,str("intersection"),(x+w+10,y+h),0,0.3,(255,0,0))


x, y, w, h = 83, 440, 229, 614
cv2.rectangle(imKnown,(x,y),(x+w,y+h),(0,0,255),2)
cv2.putText(imKnown,str("final"),(x+w+10,y+h),0,0.3,(255,0,0))


cv2.imshow('Known', imKnown)
cv2.waitKey(0)
cv2.destroyAllWindows()