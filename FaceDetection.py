import numpy as np
import json
import cv2
from os import listdir

if __name__ == "__main__":

    # face_cascade is the classifier from opencv to detect face.
    # haarcascade_frontalface_default.xml is the data provided by the opencv.
    face_cascade = cv2.CascadeClassifier('Model_Files/haarcascade_frontalface_default.xml')

    # images are all the images from the Test folder.
    images = listdir('Test folder/images')

    # json_list is to store json dictionary.
    json_list = []
    c = 1
    for img in images:
        # First: reads the image data.
        data = cv2.imread('Test folder/images/'+img)
        gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        # Then: run the face detection from the opencv library.
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            # Retrieve the x, y, width, and height. Store into the json_list.
            json_ele = {"iname": img, "bbox": [int(x), int(y), int(w), int(h)]}
            json_list.append(json_ele)
        c += 1

    # Finally: write the results into the results.json file.
    output_json = "results.json"
    with open(output_json, 'w') as f:
        json.dump(json_list, f)
