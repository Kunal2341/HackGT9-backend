from fastapi import FastAPI
import cv2
import numpy as np
import os

app = FastAPI()
app.areas_to_tunes = {}
app.shapes = {
    -1: "random",
    0: "drum",
    1: "piano",
    2: "hat" 
}
app.folder_path = "ex-images"

def intersects(box1, box2):
    """Helper function for get_coordinates"""

    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[1] > box2[3] or box1[3] < box2[1])

def get_coordinates(path_to_image: str):
    """Returns the coordinates of the shapes in the image"""

    LOWER_BLUE_COLOR = [25,25,0]
    UPPER_BLUE_COLOR = [255,255,255]
    CONTOUR_SIZE_RESTRICTION = 40
    BORDER_SHAPE_ADD = 15

    img = cv2.imread(path_to_image)
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array(LOWER_BLUE_COLOR) 
    upper_blue = np.array(UPPER_BLUE_COLOR)
    mask_blue = cv2.inRange(imghsv, lower_blue, upper_blue)
    #Show masked image
    # cv2.imshow("winname" , mask_blue)
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    im = np.copy(img)
    imSizeRestrict = np.copy(img)
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

    textBuffer = 5

    bufferedDimensions = []

    for shapeFinalizaed in dimensionsShapes:
        x, y, w, h = shapeFinalizaed
        bufferedDimensions.append([x - BORDER_SHAPE_ADD, y - BORDER_SHAPE_ADD,
            w + BORDER_SHAPE_ADD * 2, h + BORDER_SHAPE_ADD * 2])

    return bufferedDimensions

def get_shape(path_to_image, coordinate):
    """-1: Random, 0: Drum, 1: Piano Tile, 2: High Hat"""
    
    img = cv2.imread(path_to_image)

    return -1

def find_note(path_to_image, coordinate):
    """Return A - G note (Just return letter recognized inside shape, if none return empty string"""
    # letter recognition  in shape
    return ""

def generate_random_tune(path_to_image, coordinate):
    """After talking to rishi, generate some output given some shape"""
    
    return 'bruh'

def get_tune(path_to_image, coordinate):
    """Return instrument and note based on shape"""

    shape_idx = get_shape(path_to_image, coordinate)
    note = ''
    if shape_idx > -1:
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

    path_to_image = os.path.join(app.folder_path, "1.jpg")

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