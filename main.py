

import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras import backend as K
import time

#from google.colab.patches import cv2_imshow

# Define the custom triplet_loss function
def triplet_loss(y_true, y_pred):
    margin = 1.0
    positive_distance = tf.reduce_sum(tf.square(y_pred[:, 0:128] - y_pred[:, 128:256]), axis=1)
    negative_distance = tf.reduce_sum(tf.square(y_pred[:, 0:128] - y_pred[:, 256:384]), axis=1)
    loss = tf.maximum(positive_distance - negative_distance + margin, 0.0)
    return loss

# Load FaceNet model
#facenet_model = load_model('./facenet_keras.h5')
#model = tf.keras.applications.ResNet50(weights='imagenet')
# Load FaceNet model
facenet_model = tf.keras.models.load_model('models/facenet.pb', custom_objects={'triplet_loss': triplet_loss})
# Initialize YOLOv8 model
yolo_model = YOLO('yolov8n-face (3) (2).pt')  # Specify the path to your model here


# Load known face encodings and their names from files
known_face_encodings = np.load('./embeddings (7).npy')
known_face_names = np.load('./labels (6).npy')

# Function to get face embedding using FaceNet
def get_face_embedding(face_image):
    face_image = cv2.resize(face_image, (160, 160))
    face_image = np.expand_dims(face_image, axis=0)
    embedding = facenet_model.predict(face_image)
    return embedding[0]

# Initialize video capture
#video_path = '/content/Incredible_public_support_for_PM_Modi_in_Mysuru,_Karnataka___PM_Modi_in_Karnataka(1080p).mp4'  # Change this to your video file path
video_capture = cv2.VideoCapture(0)
#time.sleep(10)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the image from BGR color (OpenCV format) to RGB color
    rgb_frame = frame[:, :, ::-1]

    # Detect faces using YOLOv8
    yolo_results = yolo_model(frame)

    for result in yolo_results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            label = box.cls
            confidence = box.conf

            # If the detected object is a person (label 0), process further
            if label == 0:
                face_image = rgb_frame[y1:y2, x1:x2]

                # Get face embedding using FaceNet
                face_embedding = get_face_embedding(face_image)

                # Compare the face embedding with known face encodings
                name = "Unknown"
                if len(known_face_encodings) > 0:
                    distances=[]
                    for encoding in known_face_encodings:
                        distance = np.linalg.norm(known_face_encodings - face_embedding[:128])
                        distances.append(distance)
                        min_distance=np.min(distances)
                        if min_distance < 11:  # Adjust the threshold as needed
                            index = np.argmin(distances)
                            name = known_face_names[index]

                # Draw a rectangle around the face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (x1 + 6, y2 - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow("result", frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the video file
video_capture.release()
cv2.destroyAllWindows()
