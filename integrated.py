
import cv2
import numpy as np
from ultralytics import YOLO
import face_recognition
from keras.models import load_model
from wide_resnet import WideResNet  # Replace with correct import path for WideResNet

# Load models
emotion_model = load_model("model/emotion_little_vgg_2.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
age_gender_model = WideResNet(64, depth=16, k=8)()
age_gender_model.load_weights("pretrained_models/weights.28-3.73.hdf5")  # Replace with correct path
yolo_model = YOLO('yolov8n.pt')  # YOLO model for queue detection

# Load training image for face recognition
image_path = "C:/Users/saniy/Downloads/test.jpeg"  # Replace with your image path
try:
    my_image = face_recognition.load_image_file(image_path)
    my_face_encoding = face_recognition.face_encodings(my_image)[0]
except IndexError:
    print("Error: No faces found in the training image.")
    exit()

known_face_encodings = [my_face_encoding]
known_face_names = ["Akifwali [Staff]"]

# Helper function: Preprocess face for emotion, age, and gender detection
def preprocess_face_for_emotion(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=(0, -1))
    return face_img

def preprocess_face_for_age_gender(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

# Function to process queue management
def process_queue(frame):
    results = yolo_model(frame)
    detections = results[0].boxes
    count = 0

    for box in detections:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])

        if class_id == 0 and conf > 0.5:  # Class 0 is 'person'
            count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.putText(frame, f'Queue Count: {count}', (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame, count

# Capture video
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("Starting camera. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        # Get face coordinates
        top, right, bottom, left = face_location
        face_img = frame[top:bottom, left:right]

        if name == "Unknown":
            # Emotion Detection
            preprocessed_face_emo = preprocess_face_for_emotion(face_img)
            emotion_prediction = emotion_model.predict(preprocessed_face_emo)
            emotion_label = emotion_labels[np.argmax(emotion_prediction)]

            # Age and Gender Detection
            preprocessed_face_age_gender = preprocess_face_for_age_gender(face_img)
            age_gender_prediction = age_gender_model.predict(preprocessed_face_age_gender)
            gender_label = "Male" if np.argmax(age_gender_prediction[0]) == 1 else "Female"
            age_label = int(age_gender_prediction[1].dot(np.arange(0, 101)).flatten()[0])

            # Annotate face with emotion, age, and gender
            label = f"{emotion_label}, {gender_label}, {age_label}"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # Annotate face with name
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw rectangle around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Perform queue detection
    frame, people_count = process_queue(frame)

    # Display the frame
    cv2.imshow('Integrated Video Feed', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
