import cv2
import os 
import numpy as np
from matplotlib import pyplot as plt

imagesFolder = "ex-images"
example_img = os.path.join(imagesFolder, "2.jpg")

boundingBox = [[686, 1178, 59, 51], [542, 1168, 49, 61], [161, 924, 241, 215], [451, 780, 182, 176], [173, 776, 191, 185], [617, 747, 118, 165], [444, 657, 148, 162], [602, 596, 130, 162], [166, 505, 190, 218], [432, 375, 169, 184], [666, 289, 124, 169], [519, 128, 176, 169], [139, 85, 236, 243]]

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
    cv2.imshow("Cropped", croppedImg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()