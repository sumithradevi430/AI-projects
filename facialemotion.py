import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained emotion detection model
emotion_model = load_model('path_to_your_emotion_detection_model.h5')

# Define the emotions
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Function to detect and classify emotions
def detect_emotion(gray_face):
    face = cv2.resize(gray_face, (48, 48))
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=4)
    result = emotion_model.predict(face)
    emotion_label_arg = np.argmax(result)
    emotion_text = emotions[emotion_label_arg]
    return emotion_text

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces and classify emotions
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_gray = gray[y:y+h, x:x+w]
        emotion = detect_emotion(face_gray)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the frame
    cv2.imshow('Facial Emotion Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
