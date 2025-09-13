import cv2
import os
import json

# Create a directory to store the captured images
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)

face_id = input('Enter user ID: ')
name = input('Enter your name: ')

# Save user ID and name mapping to a JSON file
mapping_file = 'user_mapping.json'
if os.path.exists(mapping_file):
    with open(mapping_file, 'r') as file:
        user_mapping = json.load(file)
else:
    user_mapping = {}

user_mapping[face_id] = name
with open(mapping_file, 'w') as file:
    json.dump(user_mapping, file)

print("Capturing images. Look at the camera...")

count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(f'dataset/User.{face_id}.{count}.jpg', gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('image', frame)
        
        # Add a small delay to ensure the camera captures the frame
        cv2.waitKey(100)

    if count >= 30:
        break

print(f"\nCaptured {count} images for User ID: {face_id} - Name: {name}")

cap.release()
cv2.destroyAllWindows()