import cv2
import wiringpi ####wiringpi import
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from video import create_capture ####

USE_WEBCAM = True # If false, loads video file source

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming
cv2.namedWindow('window_frame')

####Led Setting
wiringpi.wiringPiSetup()
wiringpi.pinMode(1,1)
wiringpi.pinMode(4,1)
wiringpi.pinMode(5,1)

wiringpi.digitalWrite(1,0) ###blue
wiringpi.digitalWrite(4,0) ###red
wiringpi.digitalWrite(5,0) ###green

# Select video or webcam feed
cap = None
if (USE_WEBCAM == True):
    ####camera input setting
    import sys, getopt
    try:
        video_src = sys.argv[1]
    except:
        video_src=0
    cap=cv2.VideoCapture(video_src)######


while cap.isOpened(): # True:
    ret, bgr_image = cap.read()

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

       if emotion_text == 'angry':
            wiringpi.digitalWrite(1,1) ###blue
            wiringpi.digitalWrite(4,0) ###red
            wiringpi.digitalWrite(5,1) ###green
            color = np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
			wiringpi.digitalWrite(1,0)
			wiringpi.digitalWrite(4,1)
			wiringpi.digitalWrite(5,1)
            color = np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            wiringpi.digitalWrite(1,1) ###blue
            wiringpi.digitalWrite(4,0) ###red
            wiringpi.digitalWrite(5,0) ###green
            color = np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
			wiringpi.digitalWrite(1,0)
			wiringpi.digitalWrite(4,1)
			wiringpi.digitalWrite(5,0)
            color = np.asarray((0, 255, 255))
		elif emotion_text == 'neutral':
			wiringpi.digitalWrite(1,1)
			wiringpi.digitalWrite(4,0)
			wiringpi.digitalWrite(5,0)
			color = np.asarray((0, 255, 255))
        else:
            color = np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
