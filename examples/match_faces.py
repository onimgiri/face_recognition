import os
import face_recognition

import base64
import json


def images_to_json(folder_path):
    image_files = [f for f in os.listdir(
        folder_path) if f.endswith(('.jpg', '.png', ".JPG"))]
    image_list = []

    for image_file in image_files:
        with open(os.path.join(folder_path, image_file), 'rb') as file:
            base64_image = base64.b64encode(file.read()).decode('utf-8')
            image_info = {"filename": image_file, "base_64": base64_image}
            image_list.append(image_info)

    data = {
        "latlong": "27.939743871352466, -82.45443125436583",
        "total_faces": len(image_files),
        "estimated_crossing": 5,  # Update this as needed
        "faces_info": image_list,
    }

    with open('NEO_file.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)


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

# Call the function and print the output
print(images_to_json('output_faces'))
