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
    cv2.imshow("Cropped", croppedImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
