import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from IPython.display import SVG, Image
from livelossplot.inputs.tf_keras import PlotLossesCallback
import tensorflow as tf
import cv2
import random
import pygame
print("Tensorflow version:", tf.__version__)

#from google.colab import drive
#drive.mount("C:\\Users\\Alex E Mathew\\Desktop\\First Year Computer Project\\P\\drive.py")

for expression in os.listdir("C:\\Users\\Alex E Mathew\\Desktop\\First Year Computer Project\\P\\train"):
    print(str(len(os.listdir("C:\\Users\\Alex E Mathew\\Desktop\\First Year Computer Project\\P\\train\\" + expression))) + " " + expression + " images")

img_size = 48
batch_size = 64
datagen_train = ImageDataGenerator(horizontal_flip=True)
train_generator = datagen_train.flow_from_directory("C:\\Users\\Alex E Mathew\\Desktop\\First Year Computer Project\\P\\train",
                                                    target_size=(img_size,img_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)
datagen_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator = datagen_validation.flow_from_directory("C:\\Users\\Alex E Mathew\\Desktop\\First Year Computer Project\\P\\train",
                                                    target_size=(img_size,img_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)

# Initialising the CNN
model = Sequential()
# 1 - Convolution
model.add(Conv2D(64,(3,3), padding='same', input_shape=(48, 48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# 2nd Convolution layer
model.add(Conv2D(128,(5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# 3rd Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# 4th Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# Flattening
model.add(Flatten())
# Fully connected layer 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
# Fully connected layer 2nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(7, activation='softmax'))
opt = Adam(lr=0.0005)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
Image('model.png',width=400, height=200)



#%%time
epochs = 15
steps_per_epoch = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=0.00001, mode='auto')
checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy',
                             save_weights_only=True, mode='max', verbose=1)
callbacks = [PlotLossesCallback(), checkpoint, reduce_lr]
history = model.fit(
    x=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data = validation_generator,
    validation_steps = validation_steps,
    callbacks=callbacks
)



model_json = model.to_json()
model.save_weights('model_weights.h5')
with open("model.json", "w") as json_file:
    json_file.write(model_json)

from tensorflow.keras.models import model_from_json
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

import random
import cv2
import pygame
import time

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
