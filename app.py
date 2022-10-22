from fastapi import FastAPI

app = FastAPI()
app.areas_to_tunes = []
app.shapes = {
    -1: "random",
    0: "drum",
    1: "piano",
    2: "hat" 
}

def get_coordinates(path_to_image: str):
    """Returns the coordinates of the shapes in the image"""

    return []

def get_shape(path_to_image, coordinate):
    """-1: Random, 0: Drum, 1: Piano Tile, 2: High Hat"""

    return -1

def find_note(path_to_image, coordinate):
    """Return A - G note (Just return letter recognized inside shape, if none return empty string"""

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

@app.post("/update/{path_to_image}")
def update_mapping(path_to_image: str):
    """Updates the areas_to_tunes"""

    coordinates = get_coordinates(path_to_image)
    temp = []

    for coordinate in coordinates:
        temp[coordinate] = get_tune(coordinate)

    app.areas_to_tunes = temp

@app.get("/tune/{coordinate}")
def get_tune(coordinate):
    """Returns the tune for the coordinate clicked"""
    
    return app.areas_to_tunes

@app.get("/mapping")
def get_mapping():
    """Returns the areas_to_tunes. For testing"""

    return app.areas_to_tunes