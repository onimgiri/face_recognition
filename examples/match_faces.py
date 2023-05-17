import os
import face_recognition


def find_matching_faces(input_image_path, face_folder_path):
    # Load the input image
    input_image = face_recognition.load_image_file(input_image_path)
    input_face_encodings = face_recognition.face_encodings(input_image)

    if len(input_face_encodings) == 0:
        print(f"No faces found in input image {input_image_path}")
        return
    else:
        # Taking first face if there are multiple faces
        input_face_encoding = input_face_encodings[0]

    # Iterate over the images in the face folder
    for filename in os.listdir(face_folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load each face image
            face_image_path = os.path.join(face_folder_path, filename)
            face_image = face_recognition.load_image_file(face_image_path)
            face_encodings = face_recognition.face_encodings(face_image)

            if len(face_encodings) == 0:
                print(f"No faces found in image {face_image_path}")
            else:

                for face_encoding in face_encodings:
                    # Check if the input face matches the current face
                    results = face_recognition.compare_faces(
                        [face_encoding], input_face_encoding, tolerance=0.50)

                    if results[0]:  # If the input face matches the current face
                        print(
                            f"Input image matches face image {face_image_path}")
                        return  # If you want to find all matching images, remove this line

    print("No matching faces found.")


# Call the function with an input image and a face folder
find_matching_faces("test_stock.JPG", "output_faces")
