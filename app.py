from fastapi import FastAPI, Request
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
from music import playSound
from pydantic import BaseModel, validator
import shapely
from sympy import Point, Polygon
from fastapi.middleware.cors import CORSMiddleware


class JSCooordinate(BaseModel):
    x: float
    y: float

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
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_temp(path_to_image:str):

    img = cv2.imread(path_to_image)
    final = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)[1]

    cv2.imshow("bounding_box", thresh)
    cv2.waitKey(0)

    result = img.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        if (w * h) > 1000:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # print("x,y,w,h:", x, y, w, h)
            final.append([x, y, w, h])

    # show thresh and result
    cv2.imshow("bounding_box", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return final

def get_coordinates(path_to_image: str):
    """Returns the coordinates of the shapes in the image"""

    
    print(path_to_image)

    LOWER_BLUE_COLOR = [0,0,0]
    UPPER_BLUE_COLOR = [159,101,128]
    CONTOUR_SIZE_RESTRICTION = 40
    BORDER_SHAPE_PERCENT = 0.02
    imgO = cv2.imread(path_to_image)
    togetherBuffer = 10

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
    cleanedDimensionShapes = []

    # Checks each value with all others in arrary to see intersection, removes smaller array
    for shape in dimensionsShapes:
        for shapeCheck in dimensionsShapes:


            if intersects((max(shape[0]-togetherBuffer,0),max(shape[1]-togetherBuffer,0),shape[2]+shape[0]+togetherBuffer,shape[3]+shape[1]+togetherBuffer), 
                        (max(shapeCheck[0]-togetherBuffer,0),max(shapeCheck[1]-togetherBuffer,0),shapeCheck[2]+shapeCheck[0]+togetherBuffer,shapeCheck[3]+shapeCheck[1]+togetherBuffer)) and shape != shapeCheck:
                
                thePoints = []
                thePoints.append((shape[0], shape[1]))
                thePoints.append((shape[0] + shape[2], shape[1]))
                thePoints.append((shape[0], shape[1] + shape[3]))
                thePoints.append((shape[0] + shape[2], shape[1] + shape[3]))
                thePoints.append((shapeCheck[0], shapeCheck[1]))
                thePoints.append((shapeCheck[0] + shapeCheck[2], shapeCheck[1]))
                thePoints.append((shapeCheck[0], shapeCheck[1] + shapeCheck[3]))
                thePoints.append((shapeCheck[0] + shapeCheck[2], shapeCheck[1] + shapeCheck[3]))
                

                x, y = min(thePoints)
                if max(thePoints) == thePoints[3]:
                    w = shape[2]
                    h = shape[3]
                elif max(thePoints) == thePoints[7]:
                    w = shapeCheck[2]
                    h = shapeCheck[3]
                else:
                    raise Exception("Ask help")
                print(min(thePoints))
                print(max(thePoints))
                
                cleanedDimensionShapes.append([x,y,w,h])

    bufferedDimensions = []
    savingImg = imgO
    for shapeFinalizaed in cleanedDimensionShapes:
        border = max(int(w * imgO.shape[1] / img.shape[1] * BORDER_SHAPE_PERCENT), 
                    int(h * imgO.shape[0] / img.shape[0] * BORDER_SHAPE_PERCENT))
        x, y, w, h = shapeFinalizaed 
        x = max(int(x * imgO.shape[1] / img.shape[1] - border),0)
        y = max(int(y * imgO.shape[0] / img.shape[0] - border),0)
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
        p1 = Polygon([(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])
        p2 = Polygon([(x1,y1), (x1,y2), (x2,y1), (x2, y2)])
        if (p1.intersects(p2).area > 10): 
            return text

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
        note = "" # find_note(path_to_image, coordinate)
    else:
        note = 0 #generate_random_tune(path_to_image, coordinate)

    return {
        "instrument": app.shapes[shape_idx],
        "note": note
    }

def collides(coordinate, coor):
    x1 = coor[0]
    y1 = coor[1]
    w = coor[2]
    h = coor[3]

    x = float(coordinate["x"])
    y = float(coordinate["y"])

    return x > x1 and x < (x1 + w) and y > y1 and y < (y1 + h)

@app.post("/update/{file_name}")
def update_mapping(file_name: str):
    """Updates the areas_to_tunes"""

    path_to_image = os.path.join(r"C:\Users\Saurinya\Downloads", file_name)

    img = cv2.imread(path_to_image)
    cv2.imshow("img", img)
    cv2.waitKey(0)

    coordinates = get_temp(path_to_image)
    temp = {}

    for coordinate in coordinates:

        x1 = coordinate[0]
        y1 = coordinate[1]
        w = coordinate[2]
        h = coordinate[3]

        x2 = x1 + w
        y2 = y1 + h

        roi = img[y1:y2, x1:x2]

        cv2.imshow("roi", roi)
        cv2.waitKey(0)
        temp[tuple(coordinate)] = get_tune(path_to_image, coordinate)
    cv2.destroyAllWindows()
    app.areas_to_tunes = temp

    return {"result": coordinates}

@app.post("/tune/{points}")
async def tune(points: str):
    """Plays the tune for the coordinate clicked"""

    coordinate = {
        "x": points.split('_')[0],
        "y": points.split('_')[0],
    }
    # print(app.areas_to_tunes)
    for coor in app.areas_to_tunes:
        # print(coor, collides(coordinate, coor))
        if (collides(coordinate, coor)):
            area = app.areas_to_tunes[coor]
            playSound(area["instrument"], area["note"])
            return area["instrument"]

@app.get("/mapping")
def get_mapping():
    """Returns the areas_to_tunes. For testing"""

    return app.areas_to_tunes

@app.get("/shapes/{file_name}")
def get_shapes(file_name):
    """Returns the tune for the coordinate clicked"""
    
    # path_to_image = os.path.join(r"C:/Users/saksh/Downloads", file_name)
    # coordinates = get_coordinates(path_to_image)
    return app.areas_to_tunes.keys()
    return coordinates
