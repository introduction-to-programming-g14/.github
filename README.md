# Title of the project: FACIAL EMOTIONS RECOGNIZATION CODE
## Team members:
 Alex E Mathew - 102201010
 Ch.V.N.L.Chaitanya - 122201004
 Thamballa Sindhu - 102201001
## Introduction:
 Emotion detection and subsequent playing of appropriate music is an interdisciplinary project that combines the computer vision, audio processing and machine learning. The project aims to incorporate these techniques for the betterment of our daily lives that include entertainment and healthcare.  
 The application of this project is very diverse. Using this, we can increase the interactivity between computer and user. Consider the case of selecting music. This project helps the user to listen music depending upon their mood. Also, the project be used in healthcare like stress reduction.  
 However, it is essential to consider ethical considerations and user privacy when implementing such systems. Consent and proper data handling practices should be followed to ensure user trust and protect sensitive information.  
### Methodology:
The approach and methodology employed in the provided code can be summarized as follows:  

* Data Preparation: The code uses image data for training and validation. The images are loaded using the `ImageDataGenerator` class from TensorFlow, which provides various data augmentation techniques like horizontal flipping. The data is preprocessed to convert it to grayscale and resize it to a specific size (48x48 pixels) required by the model.  

* Convolutional Neural Network (CNN) Architecture: The code defines a CNN model using the Keras Sequential API. The model consists of multiple convolutional layers with different filter sizes, followed by batch normalization, activation functions (such as ReLU), max pooling, and dropout layers for regularization. The model is designed to learn hierarchical features from the input images.  

* Model Training: The model is compiled with an appropriate optimizer (Adam), loss function (categorical cross-entropy), and evaluation metric (accuracy). The training process is carried out using the `fit()` function, which takes the prepared data generators, the number of epochs, and other parameters. During training, the model learns to classify facial expressions based on the provided labeled data.  

* Model Evaluation: The model's performance is evaluated on the validation set after each epoch. The evaluation metrics include loss and accuracy. Additionally, callbacks such as `ModelCheckpoint` and `ReduceLROnPlateau` are used to save the best model weights and adjust the learning rate during training.  

* Model Saving: After training, the trained model is saved in both JSON and HDF5 formats. The model architecture is saved in a JSON file, while the learned weights are saved in an HDF5 file. This allows the model to be loaded and used later without retraining.  

* Emotion Prediction: The code includes a `FacialExpressionModel` class that loads the saved model and weights. This class provides a method `predict_emotion()` that takes an input image and predicts the corresponding emotion using the loaded model. The class also defines a list of emotion labels for mapping the model's output predictions.  

* Real-time Emotion Detection and Music Playback: The code utilizes the `VideoCamera` class, which captures video frames using OpenCV. It applies facial detection using the Haar cascade classifier and then uses the `FacialExpressionModel` class to predict emotions from the detected faces. If a specific emotion (e.g., "Sad" or "Angry") is detected, it randomly selects a song from the defined playlist and plays it using the’ Pygame ‘library. The music playback is controlled based on the presence or absence of the detected emotions.  
 This approach combines computer vision techniques, deep learning, and music playback to achieve real-time emotion detection and appropriate music playback based on the detected emotions.
 #### ABOUT USED LIBRARIES:
 keras:  Keras allows us to define and train models easily, making it a popular choice for deep learning tasks.It is mainly used for training       networks. Keras is used for creating deep models that can be productized on smartphones.
 OpenCV: OpenCV (Open Source Computer Vision Library) is a popular computer vision library that provides a wide range of functions and algorithms   for image and video processing.It provides various methods for detecting, recognizing, and analyzing faces and facial expressions in our         code.It is also used for preprocessing the image data such as inverting,resizing,cropping...etc.
 numpy: It is a fundamental library for scientific computing in python.In our project NumPy is also used for manipulating and processing the       image data as arrays.
 tensorflow: It provides various tools and functions for building, training, and deploying complex models on different platforms and devices. In   our project tensorflow is used for implementing the core algorithms and computations involved in facial emotion recognition.
 
