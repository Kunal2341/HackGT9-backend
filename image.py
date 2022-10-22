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
im = np.copy(img)

sizedContours = []

for c in contours:
    if (cv2.contourArea(c) > 40): 
        sizedContours.append(c)
        
        """
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        print(rect)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(im,str(cv2.contourArea(c)),(x+w+10,y+h),0,0.3,(255,0,0))
        """
print("Detected " +str(len(sizedContours)) + " shapes.")


for con in sizedContours:
    #Possible Clean
    rect = cv2.boundingRect(c)
    x,y,w,h = rect
    print(rect)
    for c in sizedContours:
        Trect = cv2.boundingRect(c)
        Tx,Ty,Tw,Th = rect
        if intersects((x,y,w+x,h+y), (Tx,Ty,Tw+Tx,Th+Ty)):
            newX = min(x, Tx)
            newY = min(y, Ty)
            newW = x + w + Tw - abs(Tx - (x + w))
            newH = y + h + Th - abs(Ty - (y + h))
            cv2.rectangle(im,(newX,newY),(newX+newW,newY+newH),(0,255,0),2)
            cv2.putText(im,str(cv2.contourArea(c)),(newX+newW+10,newY+newH),0,0.3,(255,0,0))
        else:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(im,str(cv2.contourArea(c)),(x+w+10,y+h),0,0.3,(255,0,0))

cv2.drawContours(im, contours, -1, (255, 0, 0), 1)


cv2.imshow('Test Image Between 2 colors', im)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (0,0), sigmaX=100, sigmaY=100)
divide = cv2.divide(gray, blur, scale=255)
thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Gausian Blur Something', morph)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


"""
gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),5)
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# using a findContours() function
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

i = 0
# list for storing names of shapes
for contour in contours:
    # here we are ignoring first counter because 
    # findcontour function detects whole image as shape
    if i == 0:
        i = 1
        continue
    # cv2.approxPloyDP() function to approximate the shape
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

    if not (cv2.contourArea(approx) < 10):
        print(len(approx))
        # using drawContours() function
        cv2.drawContours(img, [contour], 0, (0, 0, 255), 1)
        # finding center point of shape
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
        # putting shape name at center of each shape
        if len(approx) == 3:
            cv2.putText(img, 'Triangle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif len(approx) == 4:
            cv2.putText(img, 'Quadrilateral', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif len(approx) == 5:
            cv2.putText(img, 'Pentagon', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif len(approx) == 6:
            cv2.putText(img, 'Hexagon', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(img, 'circle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  

# displaying the image after drawing contours
cv2.imshow('shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""