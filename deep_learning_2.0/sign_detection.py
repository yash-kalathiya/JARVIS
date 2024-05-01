import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from Speak import SpeakWindow
import streamlit as st

def SignDetection(box):
    detector = HandDetector(maxHands=1)
    classifier = Classifier("keras_model.h5",
                            "labels.txt")
    offset = 20
    imgSize = 300
    labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
            "w", "x", "y", "z", " ", "del"]
    copy_last_word = ""
    output_sentence = ""
    prev_prediction = ""
    prev_prediction_count = 0
    frame_count = 0
    del_count = 0
    ready_for_speech = False
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('http://192.168.221.84:8000/stream.mjpg')

    # Flag to indicate if a sentence is ready for speech conversion
    image_placeholder = st.empty()
    button = st.button("Stop",key='stop_sign')
    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            # Check if imgCrop is not empty
            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                imgCropShape = imgCrop.shape
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                if labels[index] == "del":
                    del_count += 1
                    if del_count >= 7:
                        if len(output_sentence) > 0:
                            output_sentence = output_sentence[:-1]
                        del_count = 0
                else:
                    del_count = 0
                    if labels[index] != prev_prediction:
                        prev_prediction_count = 0
                    else:
                        prev_prediction_count += 1

                    if prev_prediction_count >= 7:
                        output_sentence += labels[index]
                        prev_prediction_count = 0

                prev_prediction = labels[index]

                cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 0), 4)
                cv2.putText(imgOutput, output_sentence, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                # cv2.imshow("ImageWhite", imgWhite)

        image_placeholder.image(imgOutput,channels="BGR", use_column_width=True)
        if button:
            return

        frame_count += 1
        if frame_count == 7:
            frame_count = 0

        if output_sentence and ready_for_speech:
            words = output_sentence.split()
            last_word = words[-1]
            copy_last_word = last_word
            SpeakWindow(last_word.strip(),box)
            ready_for_speech = False
            last_word = ""

        if output_sentence and (output_sentence[-1] == " "):
            words = output_sentence.split()
            last_word = words[-1]
            if last_word and (last_word == copy_last_word):
                ready_for_speech = False
            else:
                ready_for_speech = True
        