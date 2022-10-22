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
    """-1: Random, 0: Drum, 1: Piano Tile, 2: High Hat"""
    
    #img = cv2.imread(path_to_image)

    x1 = coordinate[0]
    y1 = coordinate[1]
    w = coordinate[2]
    h = coordinate[3]

    x2 = x1 + w
    y2 = y1 + h

    #roi = img[y1:y2, x1:x2]

    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file("../hackgt-366316-0c3450bdab27.json")
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient(credentials=credentials)

    with io.open(path_to_image, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    for text in texts:
        p1 = Polygon([(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])
        p2 = Polygon([(x1,y1), (x2,y1), (x1,y2), (x2,y2)])
        if(p1.intersects(p2)):
            return(text.description)
            #Change what too return 
    if response.error.message:
        return -2

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