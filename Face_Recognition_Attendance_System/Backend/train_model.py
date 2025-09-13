import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Function to get the images and label data
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for image_path in image_paths:
        gray_img = Image.open(image_path).convert('L')
        img_numpy = np.array(gray_img, 'uint8')
        id_ = int(os.path.split(image_path)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id_)

    return face_samples, ids

print("\nTraining faces. It will take a few seconds. Wait ...")
faces, ids = get_images_and_labels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer.yml
recognizer.write('trainer.yml')

print("\nModel trained and saved in trainer.yml")