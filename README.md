Face Recognition Attendance System

This project is a Face Recognition based Attendance System that captures faces, trains a model, and marks attendance automatically when a known face is recognized.


Features

Capture images of new users
Trains a face recognition model
Recognize faces in real time using a webcam
Record attendance automatically into an Excel file


Installation

Clone the repository

git clone https://github.com/dimple624/Face_Recognition_Attendance_System.git
cd Face_Recognition_Attendance_System/Backend


Install dependencies

pip install -r requirements.txt


Usage

Capture Images

python capture_images.py

This will store face images into the dataset/ folder.

Train the Model

python train_model.py

Recognize and Mark Attendance

python recognize_faces.py

Recognized faces will be logged into attendance.xlsx.


Requirements:
opencv-python
face-recognition
numpy
pandas
