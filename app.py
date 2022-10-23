from fastapi import FastAPI
import cv2
import numpy as np
import os
from utils import perform_cleanup, intersects
from keras.models import load_model
from music import playSound
from google.oauth2 import service_account
from google.cloud import vision
import io
import base64
import json

app = FastAPI()
app.areas_to_tunes = {}
app.shapes = {
    0: "random",
    1: "drum",
    2: "piano",
    3: "hat" 
}
app.folder_path = "ex-images"
app.model = load_model("./shapes.model.01.h5")

def get_coordinates(path_to_image: str):
    """Returns the coordinates of the shapes in the image"""
    LOWER_BLUE_COLOR = [25,25,0]
    UPPER_BLUE_COLOR = [255,255,255]
    CONTOUR_SIZE_RESTRICTION = 40
    BORDER_SHAPE_PERCENT = 0.05
    imgO = cv2.imread(path_to_image)
    
    # converting image into grayscale image
    scale_percent = 30
    width = int(imgO.shape[1] * scale_percent / 100)
    height = int(imgO.shape[0] * scale_percent / 100)
    dsize = (width, height)
    img = cv2.resize(imgO, dsize)

    # Get image from between 2 main colors 
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array(LOWER_BLUE_COLOR) 
    upper_blue = np.array(UPPER_BLUE_COLOR)
    mask_blue = cv2.inRange(imghsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    im = np.copy(img)
    dimensionsShapes = []

    for c in contours:
        if (cv2.contourArea(c) > CONTOUR_SIZE_RESTRICTION): 
            rect = cv2.boundingRect(c)
            x,y,w,h = rect
            dimensionsShapes.append([x,y,w,h])
    
    # Checks each value with all others in arrary to see intersection, removes smaller array
    for shape in dimensionsShapes:
        for shapeCheck in dimensionsShapes:
            if intersects((shape[0],shape[1],shape[2]+shape[0],shape[3]+shape[1]), (shapeCheck[0],shapeCheck[1],shapeCheck[2]+shapeCheck[0],shapeCheck[3]+shapeCheck[1])) and shape != shapeCheck:
                if shape[2]*shape[3] > shapeCheck[2]*shapeCheck[3]:
                    dimensionsShapes.remove(shapeCheck)
                else:
                    dimensionsShapes.remove(shape)

    bufferedDimensions = []
    savingImg = imgO
    for shapeFinalizaed in dimensionsShapes:
        border = max(int(w * imgO.shape[1] / img.shape[1] * BORDER_SHAPE_PERCENT), 
                    int(h * imgO.shape[0] / img.shape[0] * BORDER_SHAPE_PERCENT))
        x, y, w, h = shapeFinalizaed 
        x = int(x * imgO.shape[1] / img.shape[1] - border)
        y = int(y * imgO.shape[0] / img.shape[0] - border)
        w = int(w * imgO.shape[1] / img.shape[1]) + border * 2
        h = int(h * imgO.shape[0] / img.shape[0]) + border * 2

        bufferedDimensions.append([x,y,w,h])
    return bufferedDimensions

def get_shape(path_to_image, coordinate):
    """0: Random, 1: Drum, 2: Piano Tile, 3: High Hat"""
    
    img = cv2.imread(path_to_image)

    x1 = coordinate[0]
    y1 = coordinate[1]
    w = coordinate[2]
    h = coordinate[3]

    x2 = x1 + w
    y2 = y1 + h

    roi = img[y1:y2, x1:x2]

    img = perform_cleanup(roi)
    prediction = np.argmax(app.model.predict(img))
    return prediction

def find_note(path_to_image, coordinate):
    """Return A - G note (Just return letter recognized inside shape, if none return empty string"""
    # letter recognition  in shape
    img = cv2.imread(path_to_image)

    x1 = coordinate[0]
    y1 = coordinate[1]
    w = coordinate[2]
    h = coordinate[3]

    x2 = x1 + w
    y2 = y1 + h

    roi = img[y1:y2, x1:x2]

    credentials = service_account.Credentials.from_service_account_file("../hackgt-366316-0c3450bdab27.json")
    client = vision.ImageAnnotatorClient(credentials=credentials)

    image_string = cv2.imencode('.jpg', roi)[1]
    image_string = base64.b64encode(image_string).decode()

    image = vision.Image(content=bytes(image_string))

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        return ""

    return ""

def generate_random_tune(path_to_image, coordinate):
    """After talking to rishi, generate some output given some shape"""
    
    return 'bruh'

def get_tune(path_to_image, coordinate):
    """Return instrument and note based on shape"""

    shape_idx = get_shape(path_to_image, coordinate)
    note = ''
    if shape_idx > 0:
        note = find_note(path_to_image, coordinate)
    else:
        note = generate_random_tune(path_to_image, coordinate)

    return {
        "instrument": app.shapes[shape_idx],
        "note": note
    }

@app.post("/update/{file_name}")
def update_mapping(file_name: str):
    """Updates the areas_to_tunes"""

    path_to_image = os.path.join(app.folder_path, file_name)

    coordinates = get_coordinates(path_to_image)
    temp = {}

    for coordinate in coordinates:
        temp[tuple(coordinate)] = get_tune(path_to_image, coordinate)

    app.areas_to_tunes = temp

@app.get("/tune/{coordinate}")
def tune(coordinate):
    """Returns the tune for the coordinate clicked"""
    
    return app.areas_to_tunes

@app.get("/mapping")
def get_mapping():
    """Returns the areas_to_tunes. For testing"""

    return app.areas_to_tunes

@app.get("/shapes/{file_name}")
def get_shapes(file_name):
    """Returns the tune for the coordinate clicked"""
    
    path_to_image = os.path.join(app.folder_path, file_name)
    coordinates = get_coordinates(path_to_image)
    return coordinates