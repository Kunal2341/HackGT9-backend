from google.oauth2 import service_account
import time
start_time = time.time()
credentials = service_account.Credentials.from_service_account_file("../hackgt-366316-0c3450bdab27.json")

def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient(credentials=credentials)



    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    print(content)
    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    for text in texts:
        p1 = Polygon([(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])
        p2 = Polygon([(0,1), (1,0), (1,1)])
        print(p1.intersects(p2))
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))




    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))




detect_text("letterImageDetect.png")
print(time.time() - start_time)