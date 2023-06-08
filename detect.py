from tensorflow.keras.models import model_from_json
import random
import cv2
import pygame
import time
import numpy as np

class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust",
                    "Fear", "Happy",
                    "Neutral", "Sad",
                    "Surprise"]
    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()
    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]


facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX



pygame.mixer.init()  # Initialize the mixer module

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.emotion_detected = False  # Flag to track if angry or sad emotion is detected
        self.music_playing = False  # Flag to track if music is currently playing
        self.emotion_timer = None  # Timer to track elapsed time without emotion detection

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            if pred == "Sad" or pred == "Angry":
                print("Emotion detected:", pred)
                self.emotion_detected = True
                self.emotion_timer = None  # Reset the timer
                break

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        if self.emotion_detected and not self.music_playing:
            # Play a random song from the playlist
            print("Playing a random song...")
            song = random.choice(playlist)
            pygame.mixer.music.load(song)
            pygame.mixer.music.play()
            self.music_playing = True
        elif not self.emotion_detected and self.music_playing:
            if self.emotion_timer is None:
                self.emotion_timer = time.time()
            else:
                elapsed_time = time.time() - self.emotion_timer
                if elapsed_time >= 10 or cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopping the music...")
                    pygame.mixer.music.stop()
                    self.music_playing = False
                    self.emotion_timer = None


        return fr

# ... Rest of the code remains the same ...

# Playlist of songs
folder_path = "C:/Users/Alex E Mathew/Desktop/First Year Computer Project/P/Music/"
playlist = [
    folder_path + "The Beatles - Hey Jude.mp3",
    folder_path + "Journey - Don't Stop Believin' (Live 1981_ Escape Tour -2022 HD Remaster).mp3",
    folder_path + "Ee Mizhikalen- Ormayundo Ee Mukham  Vineet Sreenivasan Namitha Pramod Full song HD video.mp3",
    folder_path + "_Kabira Full Song_ Yeh Jawaani Hai Deewani  Pritam  Ranbir Kapoor, Deepika Padukone.mp3",
    folder_path + "_Tum Hi Ho_ Aashiqui 2 Full Song With Lyrics  Aditya Roy Kapur, Shraddha Kapoor.mp3"
]

try:
    camera = VideoCamera()
    while True:
        frame = camera.get_frame()
        cv2.imshow('Facial Expression Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping the program...")
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping the music...")
            pygame.mixer.music.stop()
            self.music_playing = False
            self.emotion_timer = None

except SystemExit:
    pass
