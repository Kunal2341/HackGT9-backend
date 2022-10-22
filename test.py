from starlette.testclient import TestClient
import cv2
import numpy as np
import os
from app import app

client = TestClient(app)

def test_update():
    resp = client.post("/update/2.jpg")

def perform_detection(imageread):

    #converting the input image to grayscale image using cvtColor() function
    imagegray = cv2.cvtColor(imageread, cv2.COLOR_BGR2GRAY)
    #using threshold() function to convert the grayscale image to binary image
    _, imagethreshold = cv2.threshold(imagegray, 245, 255, cv2.THRESH_BINARY_INV)

    #finding the contours in the given image using findContours() function
    imagecontours, _ = cv2.findContours(imagethreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #for each of the contours detected, the shape of the contours is approximated using approxPolyDP() function and the contours are drawn in the image using drawContours() function
    # print(imagecontours)
    for count in imagecontours:
        epsilon = 0.01 * cv2.arcLength(count, True)
        approximations = cv2.approxPolyDP(count, epsilon, True)
        cv2.drawContours(imageread, [approximations], 0, (0), 3)
        #the name of the detected shapes are written on the image
        i, j = approximations[0][0] 
        if len(approximations) == 3:
            cv2.putText(imageread, "Triangle", (i, j), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
        elif len(approximations) == 4:
            cv2.putText(imageread, "Rectangle", (i, j), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
        elif len(approximations) == 5:
            cv2.putText(imageread, "Pentagon", (i, j), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
        elif 6 < len(approximations) < 15:
            cv2.putText(imageread, "Ellipse", (i, j), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
        else:
            cv2.putText(imageread, "Circle", (i, j), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
    #displaying the resulting image as the output on the screen
    cv2.imshow(str(len(approximations)), imageread)
    cv2.waitKey(0)

def perform_prediction(img):

    imagegray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, imagethreshold = cv2.threshold(imagegray, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('bruh', imagethreshold)
    cv2.waitKey(0)

def show_shapes():
    file_name = "2.jpg"

    coords = client.get("/shapes/" + file_name).json()
    # print(coords.json())
    img = cv2.imread("ex-images/" + file_name)

    for shape in coords[2:]:
        x1 = shape[0]
        y1 = shape[1]
        w = shape[2]
        h = shape[3]
        x2 = x1 + w
        y2 = y1 + h
        croppedImg = img[y1:y2, x1:x2]
        perform_prediction(croppedImg)
        # cv2.imshow('bruh', croppedImg)
        # cv2.waitKey(0)

    cv2.destroyAllWindows() 

# test_update()
show_shapes()
# perform_prediction(cv2.imread("ex-images/2.jpg"))


# import numpy as np
# import cv2 as cv

# img = cv.imread('ex-images/2.jpg')
# imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(imgray, 127, 255, 0)
# contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(img, contours, -1, (0,255,0), 3)

# print(len(contours))
# cv2.imshow('Contours', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()