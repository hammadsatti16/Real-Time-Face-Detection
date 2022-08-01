import cv2


import cv2
def op_model(image):
    frontal_face_cascPath = "./opencv-model/haarcascade_frontalface_default.xml"
    eyes_cascPath = "./opencv-model//haarcascade_eye (1).xml"
    smile_cascPath = "./opencv-model/haarcascade_smile.xml"
    # Create the haar cascade
    frontal_face_casc = cv2.CascadeClassifier(frontal_face_cascPath)
    eyes_casc = cv2.CascadeClassifier(eyes_cascPath)
    smile_casc = cv2.CascadeClassifier(smile_cascPath)
    img = cv2.imread(image)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = frontal_face_casc.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eyes_casc.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        smile = smile_casc.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in smile:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return img