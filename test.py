def intersects(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[1] > box2[3] or box1[3] < box2[1])

def get_coordinates(path_to_image: str):
    import cv2
    import numpy as np
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
print(get_coordinates("ex-images/1.jpg"))