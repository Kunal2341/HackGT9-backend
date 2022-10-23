from starlette.testclient import TestClient
import cv2
import numpy as np
import os
from app import app
from numpy import savetxt
from utils import perform_cleanup
from keras.models import load_model

client = TestClient(app)

def test_update():
    resp = client.post("/update/2.jpg")

def show_shapes():
    file_name = "2.jpg"

    coords = client.get("/shapes/" + file_name).json()
    # print(coords.json())
    img = cv2.imread("ex-images/" + file_name)
    model = load_model("./shapes.model.01.h5")
    shapes = {
        0: "random",
        1: "drum",
        2: "piano",
        3: "hat" 
    }

    for shape in coords[2:]:
        x1 = shape[0]
        y1 = shape[1]
        w = shape[2]
        h = shape[3]
        x2 = x1 + w
        y2 = y1 + h
        croppedImg = img[y1:y2, x1:x2]
        roi = perform_cleanup(croppedImg)
        shape = model.predict(roi)

        prediction = np.argmax(shape)
        cv2.imshow(shapes[prediction], croppedImg)
        cv2.waitKey(0)

    cv2.destroyAllWindows() 

# test_update()
# show_shapes()
# perform_prediction(cv2.imread("ex-images/2.jpg"))