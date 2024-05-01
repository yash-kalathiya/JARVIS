import pyttsx3
import warnings
import streamlit as st

# Suppress all warnings
warnings.filterwarnings("ignore")

def SpeakWindow(Text,box):
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    engine.setProperty('voice',voices[0].id)
    engine.setProperty('rate',190)
    engine.say(Text)
    print(f"\nJarvis  : {Text}.\n")
    box.write(f"\nJarvis  : {Text}.\n")
    engine.runAndWait()
    box.empty()
