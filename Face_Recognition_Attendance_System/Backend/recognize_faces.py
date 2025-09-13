import cv2
import numpy as np
import os
import csv
import json
from datetime import datetime

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the face recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

# Initialize and start the video capture
cap = cv2.VideoCapture(0)

# Dictionary to keep track of recognized faces
recognized_faces = {}

# Load user ID to name mapping
mapping_file = 'user_mapping.json'
if os.path.exists(mapping_file):
    with open(mapping_file, 'r') as file:
        user_mapping = json.load(file)
else:
    user_mapping = {}

# Create or open the attendance CSV file
attendance_file = 'attendance.csv'

# Check if the attendance file exists; if not, create it with headers
if not os.path.isfile(attendance_file):
    with open(attendance_file, 'w', newline='') as csvfile:
        fieldnames = ['ID', 'Name', 'Date', 'Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)

        if conf < 50:
            # If confidence is low, mark attendance
            if id_ not in recognized_faces:
                name = user_mapping.get(str(id_), "Unknown")
                print(f"ID: {id_}, Name: {name}, Confidence: {conf}")
                recognized_faces[id_] = datetime.now()
                # Log attendance
                with open(attendance_file, 'a', newline='') as csvfile:
                    fieldnames = ['ID', 'Name', 'Date', 'Time']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    now = datetime.now()
                    date = now.strftime("%Y-%m-%d")
                    time = now.strftime("%H:%M:%S")
                    writer.writerow({'ID': id_, 'Name': name, 'Date': date, 'Time': time})

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()