import google.generativeai as genai
import cv2
import PIL.Image
from Listen import Listen
from Speak import SpeakWindow
from sign_detection import SignDetection
from face_detection import FaceDetection
import warnings
import streamlit as st

# Suppress all warnings
warnings.filterwarnings("ignore")

def gemini(question):
    if question == "":
        return "not understand"
    else:
        genai.configure(api_key='')
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(question)
        return response.text.replace("*", "").split('\n')[:8][0]

def gemini_vision(image):
    genai.configure(api_key='')
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content(['explain image', image])
    return response.text.split('\n')[:8][0]

def continuous_listen():
    while True:
        text = st.empty()
        query = Listen(text)
        if any(keyword in query for keyword in ['jarvis','hey']):
            query = query.replace("jarvis","").strip()
            query = query.replace("hey","").strip()
            
            if "front of my camera" in query:
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                cap.release()
                cv2.imwrite("captured_image.jpg", frame)
                SpeakWindow("Image captured Sucessfully",text)
                img = PIL.Image.open('captured_image.jpg')
                img_box =st.image(img, use_column_width=True)
                if ret:
                    ans = gemini_vision(img)
                    print("jarvis : ", ans)
                    SpeakWindow(ans,text)
                else:
                    print("Failed to capture image from webcam.")
                img_box.empty()
                
            elif "sign detection" in query:
                SpeakWindow("Starting sign detection",text)
                SignDetection(text)

            elif "face detection" in query:
                SpeakWindow("Starting face detection",text)
                FaceDetection(text)

            else:
                ans = gemini(query)
                print("jarvis : ", ans)
                SpeakWindow(ans,text)

            if any(keyword in query for keyword in ["bye", "exit", "turnoff", "shutdown", "shut down"]):
                st.write("Closing the program...")
                st.stop()

st.title("J.A.R.V.I.S")
if st.button("Start"):
    continuous_listen() 