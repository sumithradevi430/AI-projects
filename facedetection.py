import cv2

# Path to the Haar cascade file for frontal face detection
alg = "C:/Users/Smile/Downloads/8f51e58ac0813cb695f3733926c77f52-07eed8d5486b1abff88d7e34891f1326a9b6a6f5/haarcascade_frontalface_default.xml"

# Creating a CascadeClassifier object
haar_cascade = cv2.CascadeClassifier(cv2.samples.findFile(alg))

# Opening the default camera (index 0)
cam = cv2.VideoCapture(0)

while True:
    # Reading a frame from the camera
    _, img = cam.read()
    
    # Converting the frame to grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detecting faces in the grayscale image
    faces = haar_cascade.detectMultiScale(grayImg, scaleFactor=1.3, minNeighbors=4)
    
    # Drawing rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # Displaying the image with detected faces
    cv2.imshow("FaceDetection", img)
    
    # Waiting for a key to be pressed
    key = cv2.waitKey(10)
    
    # If the Esc key is pressed, exit the loop
    if key == ord("a"):
        break

# Releasing the camera
cam.release()

# Destroying all OpenCV windows
cv2.destroyAllWindows()




