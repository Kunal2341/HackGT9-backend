import cv2
import os 
import numpy as np
from matplotlib import pyplot as plt

def intersects(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[1] > box2[3] or box1[3] < box2[1])


LOWER_BLUE_COLOR = [0,0,0]
UPPER_BLUE_COLOR = [159, 101, 128]
CONTOUR_SIZE_RESTRICTION = 40
BORDER_SHAPE_PERCENT = 0.03


imagesFolder = "ex-images"
example_img = os.path.join(imagesFolder, "unknown.png")

#List of xy and width 
imgO = cv2.imread(example_img)

# converting image into grayscale image
scale_percent = 30
width = int(imgO.shape[1] * scale_percent / 100)
height = int(imgO.shape[0] * scale_percent / 100)
dsize = (width, height)
# resize image
img = cv2.resize(imgO, dsize)

def emptyFunction(v):
    pass



windowName = "Hello"
cv2.namedWindow(windowName)

cv2.createTrackbar('Lower - R', windowName, 0, 255, emptyFunction)
cv2.createTrackbar('Lower - G', windowName, 0, 255, emptyFunction)
cv2.createTrackbar('Lower - B', windowName, 0, 255, emptyFunction)
cv2.createTrackbar('Upper - R', windowName, 0, 255, emptyFunction)
cv2.createTrackbar('Upper - G', windowName, 0, 255, emptyFunction)
cv2.createTrackbar('Upper - B', windowName, 0, 255, emptyFunction)

# Get image from between 2 main colors 
imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array(LOWER_BLUE_COLOR) 
upper_blue = np.array(UPPER_BLUE_COLOR)
mask_blue = cv2.inRange(imghsv, lower_blue, upper_blue)
#Show masked image

def on_change(value):
    print(value)

value = 10


while(True):
    
    if cv2.waitKey(1) == 27:
        break
        
    # values of blue, green, red
    Lblue = cv2.getTrackbarPos('Lower - R', windowName)
    Lgreen = cv2.getTrackbarPos('Lower - G', windowName)
    Lred = cv2.getTrackbarPos('Lower - B', windowName)

    Ublue = cv2.getTrackbarPos('Upper - R', windowName)
    Ugreen = cv2.getTrackbarPos('Upper - G', windowName)
    Ured = cv2.getTrackbarPos('Upper - B', windowName)


    LOWER_BLUE_COLOR = [Lblue, Lgreen, Lred]
    UPPER_BLUE_COLOR = [Ublue, Ugreen, Ured]
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array(LOWER_BLUE_COLOR) 
    upper_blue = np.array(UPPER_BLUE_COLOR)
    mask_blue = cv2.inRange(imghsv, lower_blue, upper_blue)

    cv2.imshow(windowName , mask_blue)

cv2.waitKey(0)














#-------------------------------------------------------

contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

im = np.copy(img)
imSizeRestrict = np.copy(img)
imShapes = np.copy(img)

sizedContours = []
dimensionsShapes = []

shapeCountour = []

for c in contours:
    if (cv2.contourArea(c) > CONTOUR_SIZE_RESTRICTION): 
        shapeCountour.append(c)
        sizedContours.append(c)
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        dimensionsShapes.append([x,y,w,h])
        cv2.rectangle(imSizeRestrict,(x,y),(x+w,y+h),(0,100,0),2)
        cv2.putText(imSizeRestrict,str(cv2.contourArea(c)),(x+w+10,y+h),0,0.3,(255,0,0))

cv2.imshow('Known', imSizeRestrict)
cv2.waitKey(0)


# Checks each value with all others in arrary to see intersection, removes smaller array
for shape in dimensionsShapes:
    for shapeCheck in dimensionsShapes:
        if intersects((shape[0],shape[1],shape[2]+shape[0],shape[3]+shape[1]), (shapeCheck[0],shapeCheck[1],shapeCheck[2]+shapeCheck[0],shapeCheck[3]+shapeCheck[1])) and shape != shapeCheck:
            if shape[2]*shape[3] > shapeCheck[2]*shapeCheck[3]:
                dimensionsShapes.remove(shapeCheck)
            else:
                dimensionsShapes.remove(shape)

textBuffer = 5
printArray = []
ct = 1
savingImg = imgO
for shapeFinalizaed in dimensionsShapes:
    border = max(int(w * imgO.shape[1] / img.shape[1] * BORDER_SHAPE_PERCENT), 
                int(h * imgO.shape[0] / img.shape[0] * BORDER_SHAPE_PERCENT))
    print("Border width is " + str(border))

    x, y, w, h = shapeFinalizaed # Fix bro
    x = int(x * imgO.shape[1] / img.shape[1] - border)
    y = int(y * imgO.shape[0] / img.shape[0] - border)
    w = int(w * imgO.shape[1] / img.shape[1]) + border * 2
    h = int(h * imgO.shape[0] / img.shape[0]) + border * 2


    printArray.append([x,y,w,h])
    savingImg = cv2.rectangle(savingImg,(x,y),(x+w,y+h),(0,255,0),2)
    savingImg = cv2.putText(savingImg,"Shape - " + str(ct),(x+w+textBuffer,y+h),0,0.3,(255,0,0))
    ct += 1

cv2.imwrite("ResultingImg.png", savingImg)

#cv2.imshow('Known', im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

print("-"*10)
print("Detected " +str(len(dimensionsShapes)) + " shapes.")

print(printArray)



# A bunch of random other code
"""
for con in sizedContours:
    #Possible Clean
    rect = cv2.boundingRect(con)
    x,y,w,h = rect
    print(rect)
    intersect = False
    for c in sizedContours:
        Trect = cv2.boundingRect(c)
        Tx,Ty,Tw,Th = Trect
        if intersects((x,y,w+x,h+y), (Tx,Ty,Tw+Tx,Th+Ty)) and (x,y,w+x,h+y) != (Tx,Ty,Tw+Tx,Th+Ty):
            
            print(x,y,w,h)
            print(Tx,Ty,Tw,Th)
            
            newX = min(x, Tx)
            newY = min(y, Ty)
            newW = x + w + Tw - abs(Tx - (x + w))
            newH = y + h + Th - abs(Ty - (y + h))
            print(newX, newY, newW, newH)
            print()
            cv2.rectangle(im,(newX,newY),(newX+newW,newY+newH),(0,255,0),2)
            cv2.putText(im,str(cv2.contourArea(c)),(newX+newW+10,newY+newH),0,0.3,(255,0,0))
            intersect = True
    if not intersect:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(im,str(cv2.contourArea(c)),(x+w+10,y+h),0,0.3,(255,0,0))


cv2.drawContours(im, contours, -1, (255, 0, 0), 1)


cv2.imshow('Test Image Between 2 colors', im)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

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
img = imShapes
gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),5)
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# using a findContours() function
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

for contour in shapeCountour:
    # cv2.approxPloyDP() function to approximate the shape
    approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)

    if not (cv2.contourArea(approx) < 10):
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
            cv2.putText(img, 'Square', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(img, 'circle', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
  

# displaying the image after drawing contours
cv2.imshow('shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""