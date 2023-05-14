import cv2
import os
import face_recognition
import numpy as np

# Create a directory to save the faces
if not os.path.exists('output_faces'):
    os.makedirs('output_faces')

# Load the video file
input_video = cv2.VideoCapture("PXL_20230514_154510617.TS.mp4")
length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_number = 0
face_encodings_in_video = []

while True:
    # Grab a single frame of video
    ret, frame = input_video.read()
    frame_number += 1
    print("Writing frame {} / {}".format(frame_number, length))

    if not ret:
        break

     # Only process every 20th frame
    if frame_number % 60 != 0:
        continue

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if this face is in our list already
        match = None
        if len(face_encodings_in_video) > 0:
            match = face_recognition.compare_faces(
                face_encodings_in_video, face_encoding, tolerance=0.50)

        # If the face is not in video, save it to disk
        if match is None or not True in match:
            face_encodings_in_video.append(face_encoding)
            face_image = rgb_frame[top:bottom, left:right]
            cv2.imwrite(f"output_faces/face_{frame_number}.jpg",
                        cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
input_video.release()
cv2.destroyAllWindows()
