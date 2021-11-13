import keras_vggface 
import matplotlib.pyplot as plt
import numpy as np 
from mtcnn import MTCNN 
from PIL import Image

def extract_face(image, required_size=(224, 224)):
    pixels = plt.imread(image)
    detector = MTCNN()  
    results = detector.detect_faces(pixels) 
     

    x1, y1, width, height = results[0]['box'] 
    x2, y2 = x1 + width, y1 + height  
    
    face = pixels[y1:y2, x1:x2] 
    # resize the pixels to the mode size 
    image = Image.fromarray(face)
    image = image.resize(required_size) 
    face_array = np.asarray(image) 
    return face_array


