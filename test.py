from starlette.testclient import TestClient
import cv2
import numpy as np
import os
from app import app

client = TestClient(app)

def test_update():
    resp = client.post("/update/1.jpg")

    # assert resp.status_code == 200

def get_shape():
    # Coordinate: (x, y, w, h) = [1456, 303, 638, 466]
    client.get("/getshape/")

def show_shapes():
    img = cv2.imread("ex-images/1.jpg")

    x = [[803, 1561, 52, 245], [260, 1454, 521, 609], [683, 1436, 46, 37], [471, 1411, 214, 241], [1165, 1105, 552, 527], [1191, 1086, 82, 39], [1307, 1076, 69, 34], [2017, 398, 122, 43], [1456, 303, 638, 466], [210, 223, 624, 610], [1646, 111, 60, 49]]

    for shape in x:
        x1 = shape[0]
        y1 = shape[1]
        w = shape[2]
        h = shape[3]
        x2 = x1 + w
        y2 = y1 + h
        croppedImg = img[y1:y2, x1:x2]
        cv2.imshow('bruh', croppedImg)
        cv2.waitKey(0) 

    cv2.destroyAllWindows() 

test_update()
show_shapes()