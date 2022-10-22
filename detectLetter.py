from PIL import Image
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = "C:/Users/kunal/Anaconda3/envs/hackgt-9/Scripts/pytesseract.exe"


filename = 'letterImageDetect.png'
img1 = Image.open(filename)
text = pytesseract.image_to_string(img1)

print(text)