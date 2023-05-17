import cv2
import os
import face_recognition
import numpy as np
import hashlib

# Create a directory to save the faces
if not os.path.exists('output_faces'):
    os.makedirs('output_faces')

# Load the video file
input_video = cv2.VideoCapture("StockFootage.mp4")
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
    if frame_number % 20 != 0:
        continue

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        # Check if the detected face is at least 100x100
        if right - left < 75 or bottom - top < 75:
            continue

        # Check if this face is in our list already
        match = None
        # if len(face_encodings_in_video) > 0:
        #     match = face_recognition.compare_faces(
        #         face_encodings_in_video, face_encoding, tolerance=0.50)

        # If the face is not in video, save it to disk
        if match is None or not True in match:
            face_encodings_in_video.append(face_encoding)

            # Hash the face encoding
            face_encoding_hash = hashlib.sha256(
                face_encoding.tobytes()).hexdigest()

            # Save the face image, appending the hash to the filename
            face_image = rgb_frame[top:bottom, left:right]
            cv2.imwrite(f"output_faces/face_{frame_number}_{face_encoding_hash}.jpg",
                        cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))

input_video.release()
cv2.destroyAllWindows()
