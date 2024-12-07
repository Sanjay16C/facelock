from flask import Flask, render_template, Response, jsonify
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import os
import pickle
import numpy as np
import tempfile
import time

app = Flask(__name__)

# Load the saved embeddings and labels
with open('face_embeddings.pkl', 'rb') as file:
    data = pickle.load(file)
    embeddings = data['embeddings']
    labels = data['labels']

# Load the Facenet model using DeepFace
model = DeepFace.build_model("Facenet")

# Load the face detector
face_detector = cv2.CascadeClassifier('/Users/sanjayc/DEV/Projects/Faceid/haarcascade_frontalface_default.xml')

# Initialize webcam
camera = cv2.VideoCapture(0)

# Define threshold for similarity
similarity_threshold = 0.8  # Adjust this based on testing

# Variables for tracking continuous recognition
last_name = None
start_time = None
continuous_duration = 2  # 2 seconds required for "Vehicle Ready to Go!"

# Function to generate frames for video stream
def generate_frames():
    global last_name, start_time

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            detected = False  # Flag to check if any face is detected

            for (x, y, w, h) in faces:
                detected = True
                face = rgb_frame[y:y+h, x:x+w]
                try:
                    # Save the detected face region to a temporary file
                    temp_face_path = tempfile.mktemp(suffix='.jpg')
                    cv2.imwrite(temp_face_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

                    # Generate embedding for the detected face
                    face_embedding = DeepFace.represent(img_path=temp_face_path, model_name="Facenet", enforce_detection=False)[0]['embedding']

                    # Calculate cosine similarity with saved embeddings
                    similarities = cosine_similarity([face_embedding], embeddings)
                    max_similarity_idx = np.argmax(similarities)
                    max_similarity = similarities[0][max_similarity_idx]

                    # Determine the recognized name
                    name = "Unknown"
                    if max_similarity > similarity_threshold:
                        name = labels[max_similarity_idx]
                    else:
                        name = "Unknown"

                    # Continuous recognition logic
                    current_time = time.time()
                    if name == last_name and name != "Unknown":
                        if start_time is None:
                            start_time = current_time
                        elif current_time - start_time >= continuous_duration:
                            # If the name is recognized for 2 seconds
                            cv2.putText(frame, "Vehicle Ready to go!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            detected_name = name
                            print("Vehicle Ready to go!")  # This will be used in JS for redirection
                    else:
                        last_name = name
                        start_time = None

                    # Draw rectangle and text on the frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Remove the temporary face file
                    os.remove(temp_face_path)

                except Exception as e:
                    print(f"Error processing face: {e}")

            # If no face is detected or name is "Unknown"
            if not detected or (detected and name == "Unknown"):
                cv2.putText(frame, "Access Denied", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Convert the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/success')
def success():
    return render_template('success.html')

if __name__ == '__main__':
    app.run(debug=True)