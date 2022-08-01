import imghdr
import cv2
import numpy as np
import dlib
import math

def tf_model(image):
    detector = dlib.get_frontal_face_detector()
# Load the predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# read the image
    img = cv2.imread("./download.jpg")

# Convert image into grayscale
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

# Use detector to find landmarks
    faces = detector(gray)
    for face in faces:
        x1 = face.left() # left point
        y1 = face.top() # top point
        x2 = face.right() # right point
        y2 = face.bottom() # bottom point
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    # Create landmark object
        landmarks = predictor(image=gray, box=face)

    # Loop through all the points
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
        # Draw a circle
            cv2.circle(img, (x, y), 1, (255, 0, 0), 1)
    #cv2.rectangle(img, (x, y), (y, x), (0, 255, 0), 1)
    return img