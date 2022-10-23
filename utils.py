from starlette.testclient import TestClient
import cv2
import numpy as np
import os
from numpy import savetxt

def scale(im):
    desired_size = 70
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_im


def perform_cleanup(img):

    img = scale(img)
    imagegray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, imagethreshold = cv2.threshold(imagegray, 127, 255, cv2.THRESH_BINARY) # Black on white
    imagethreshold = 1.0 - (imagethreshold.astype(float))/ 255.0

    return np.array([imagethreshold])

def intersects(box1, box2):
    """Helper function for get_coordinates"""

    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[1] > box2[3] or box1[3] < box2[1])
