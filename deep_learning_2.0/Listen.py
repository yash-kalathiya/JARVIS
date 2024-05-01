import speech_recognition as sr
from googletrans import Translator
import streamlit as st
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def Listen(box):
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening....")
        box.write("Listening....")
        r.adjust_for_ambient_noise(source, duration=1)
        r.pause_threshold = 0.5
        audio = r.listen(source,0,8)

    try:
        print("Recognizing....")
        box.write("Recognizing....")
        query = r.recognize_google(audio,language="en-in")
        print(f"You : {query}")
        box.write(f"You : {query}")
    except sr.UnknownValueError:
                print("Could not understand the audio.")
                return "not understand"
    except sr.RequestError:
                print("Error connecting to Google Speech Recognition service.")
                return "not understand"

    query = str(query).lower()
    box.empty()
    return query

# 2 -  translation

def TranslationHinToEng(Text):  
    line = str(Text)
    translate = Translator()
    result = translate.translate(line)
    data = result.text
    print(f"You : {data}")
    return data

def MicExecution():
    query = Listen()
    data =TranslationHinToEng(query)
    return data

