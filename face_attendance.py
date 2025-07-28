import cv2
import face_recognition
import os
from datetime import datetime
import numpy as np


# Load known faces
known_encodings = []
known_names = []

for filename in os.listdir("known_faces"):
    img = face_recognition.load_image_file(f"known_faces/{filename}")
    encodings = face_recognition.face_encodings(img)
    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(os.path.splitext(filename)[0])

# Attendance function
def mark_attendance(name):
    with open('attendance.csv', 'a+') as f:
        f.seek(0)
        lines = f.readlines()
        names = [line.split(',')[0] for line in lines]
        if name not in names:
            now = datetime.now()
            time_str = now.strftime('%H:%M:%S')
            date_str = now.strftime('%Y-%m-%d')
            f.write(f'{name},{time_str},{date_str}\n')

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for encoding, loc in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, encoding)
        face_distances = face_recognition.face_distance(known_encodings, encoding)

        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = known_names[best_match_index]
            mark_attendance(name)
            y1, x2, y2, x1 = [v * 4 for v in loc]  # scale back
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Attendance', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
