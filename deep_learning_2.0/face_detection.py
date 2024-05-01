import cv2
import face_recognition
import os
import numpy as np
from Speak import SpeakWindow
from Listen import Listen
import warnings
import streamlit as st

# Suppress all warnings
warnings.filterwarnings("ignore")

def FaceDetection(box):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    known_faces_dir = "images\\"
    known_face_encodings = []
    known_face_names = []

    for person_folder in os.listdir(known_faces_dir):
        person_path = os.path.join(known_faces_dir, person_folder)
        if os.path.isdir(person_path):
            for file_name in os.listdir(person_path):
                img_path = os.path.join(person_path, file_name)
                known_image = face_recognition.load_image_file(img_path)
                face_locations = face_recognition.face_locations(known_image)
                if len(face_locations) > 0:
                    known_encoding = face_recognition.face_encodings(known_image)[0]
                    known_face_encodings.append(known_encoding)
                    known_face_names.append(person_folder)

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('http://192.168.221.84:8000/stream.mjpg')

    def continuous_listen():
        query = Listen(box)
        if "my name" in query:
            name = query.split("my name is")[1].capitalize()
            save_images(name)
        else:
            return
                
    def save_images(name, num_images=25, face_size=(300, 300)):
        print(f"Saving {num_images} images for {name}...")
        os.makedirs(os.path.join(known_faces_dir, name), exist_ok=True)
        count = 0

        while count < num_images:
            ret, frame = cap.read()

            if not ret:
                break

            # Convert the frame to RGB (face_recognition requires RGB images)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                # Crop the face region with desired size (300x300)
                roi_frame = rgb_frame[y:y + h, x:x + w]
                resized_face = cv2.resize(roi_frame, face_size, interpolation=cv2.INTER_AREA)

                # Convert the resized face back to BGR (for saving in color)
                bgr_resized_face = cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR)

                # Save the resized face image
                cv2.imwrite(os.path.join(known_faces_dir, name, f"{name}_{count}.jpg"), bgr_resized_face)
                count += 1

        SpeakWindow(f"{name} has been stored",box)


    unknown_count = 0  # Counter for the number of frames "Unknown" has appeared
    image_placeholder = st.empty()
    button = st.button("Stop",key='stop_face')
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]

            face_encoding = face_recognition.face_encodings(frame, [(y, x + w, y + h, x)])[0]
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                unknown_count = 0  
            else:
                unknown_count += 1  
                if unknown_count >= 10:  # Add the "Unknown" person after appearing in 5 frames
                    continuous_listen()
                    # Update known faces list after adding new face
                    for person_folder in os.listdir(known_faces_dir):
                        person_path = os.path.join(known_faces_dir, person_folder)
                        if os.path.isdir(person_path):
                            for file_name in os.listdir(person_path):
                                img_path = os.path.join(person_path, file_name)
                                known_image = face_recognition.load_image_file(img_path)
                                face_locations = face_recognition.face_locations(known_image)
                                if len(face_locations) > 0:
                                    known_encoding = face_recognition.face_encodings(known_image)[0]
                                    known_face_encodings.append(known_encoding)
                                    known_face_names.append(person_folder)
                    unknown_count = 0  # Reset the unknown count after adding to known faces

            # Draw rectangles around the face and eyes
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (x + 6, y - 6), font, 0.5, (0, 255, 100), 1)
        
        # Display the frame
        image_placeholder.image(frame,channels="BGR", use_column_width=True)
        # Break the loop when 'q' is pressed
        if button:
            break

    cap.release()
    cv2.destroyAllWindows()
