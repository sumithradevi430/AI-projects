import cv2
import os

# Path to the haarcascade file
haar_face = "C:/Users/Smile/Downloads/8f51e58ac0813cb695f3733926c77f52-07eed8d5486b1abff88d7e34891f1326a9b6a6f5/haarcascade_frontalface_default.xml"

# Directory to save the images
datasets = "/Users/Smile/Pictures/Saved Pictures"
if not os.path.exists(datasets):
    os.makedirs(datasets)

(width, height) = (130, 100)
face_cascade = cv2.CascadeClassifier(haar_face)

web = cv2.VideoCapture(0)  # Use 0 instead of 1 for the default webcam
count = 1

while count < 51:
    print(count)
    ret, im = web.read()
    if not ret:
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite("{}/{}.png".format(datasets, count), face_resize)
        count += 1
    
    cv2.imshow('opencv', im)
    key = cv2.waitKey(10)
    if key == 27:
        break

web.release()
cv2.destroyAllWindows()
