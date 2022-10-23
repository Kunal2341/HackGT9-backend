# from starlette.testclient import TestClient
# import cv2
# import numpy as np
# import os
# from app import app
# from numpy import savetxt
# from utils import perform_cleanup
# from keras.models import load_model
# import asyncio
# import threading, time

# client = TestClient(app)

# def test_update():
#     resp = client.post("/update/2.jpg")

# def show_shapes():
#     file_name = "2.jpg"

#     coords = client.get("/shapes/" + file_name).json()
#     # print(coords.json())
#     img = cv2.imread("ex-images/" + file_name)
#     model = load_model("./shapes.model.01.h5")
#     shapes = {
#         0: "random",
#         1: "drum",
#         2: "piano",
#         3: "hat" 
#     }

#     for shape in coords[2:]:
#         x1 = shape[0]
#         y1 = shape[1]
#         w = shape[2]
#         h = shape[3]
#         x2 = x1 + w
#         y2 = y1 + h
#         croppedImg = img[y1:y2, x1:x2]
#         roi = perform_cleanup(croppedImg)
#         shape = model.predict(roi)
        
#         prediction = np.argmax(shape)
#         cv2.imshow(shapes[prediction], croppedImg)
#         cv2.waitKey(0)

#     cv2.destroyAllWindows() 

# # test_update()
# # show_shapes()
# # perform_prediction(cv2.imread("ex-images/2.jpg"))

# def test_stuff():
#     client.post("/update/2.jpg")
#     json_blob = {"x": 1400.0, "y": 1200.0}
#     client.post("/tune/", json=json_blob)

# threading.Thread(target=test_stuff).start()
# test_stuff()

import cv2
import numpy as np

im = cv2.imread('./ex-images/unknown.png')
shapes = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
 
# Remember that in HSV space, Hue is color from 0..180. Red 320-360, and 0 - 30. Green is 30-100
# We keep Saturation and Value within a wide range but note not to go too low or we start getting black/gray
lower_green = np.array([0,0,0])
upper_green = np.array([360,255,50])
 
# Using inRange method, to create a mask
mask = cv2.inRange(shapes, lower_green, upper_green)
 
# We invert our mask only because we wanted to focus on the lady and not the background
# mask[mask==0] = 10
# mask[mask==255] = 0
# mask[mask==10] = 255

cv2.imshow("mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Coordinates for 2.jpg
# [585, 3130, 706, 619]
# [1453, 2550, 707, 687]
# [526, 2537, 737, 717]
# [2001, 2435, 504, 660]
# [1437, 2147, 580, 626]
# [1969, 1950, 508, 614]
# [518, 1648, 704, 797]
# [1374, 1185, 694, 744]
# [2163, 906, 528, 678]
# [1684, 380, 679, 656]
# [417, 237, 879, 903]