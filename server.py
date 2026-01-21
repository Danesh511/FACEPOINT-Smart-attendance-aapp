# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pickle
import base64
import os
import pandas as pd
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
CORS(app)  # Allows your HTML to talk to this python script

DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# --- 1. Load the AI Model ---
knn = None
def train_model():
    global knn
    try:
        # Load names and faces
        if os.path.exists(f'{DATA_DIR}/names.pkl') and os.path.exists(f'{DATA_DIR}/faces_data.pkl'):
            with open(f'{DATA_DIR}/names.pkl', 'rb') as w:
                LABELS = pickle.load(w)
            with open(f'{DATA_DIR}/faces_data.pkl', 'rb') as f:
                FACES = pickle.load(f)

            # Train KNN
            if len(LABELS) == len(FACES):
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(FACES, LABELS)
                print("✅ Model Trained Successfully")
            else:
                print("⚠️ Data Mismatch: Delete .pkl files and re-register")
    except Exception as e:
        print(f"⚠️ Training Error: {e}")

# Train on startup
train_model()

# --- Helper: Decode Base64 Image ---
def decode_image(base64_string):
    # Remove the "data:image/png;base64," prefix
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(base64_string), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# --- Route 1: Mark Attendance (Live Camera) ---
@app.route('/api/mark_attendance', methods=['POST'])
def mark_attendance():
    global knn
    if knn is None:
        return jsonify({"status": "error", "message": "System not ready (No Data)"})

    try:
        data = request.json
        image_data = data.get('image')
        location_text = data.get('location', 'Unknown')

        # 1. Process Image
        frame = decode_image(image_data)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        facedetect = cv2.CascadeClassifier(f'{DATA_DIR}/haarcascade_frontalface_default.xml')
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return jsonify({"status": "fail", "message": "No face detected"})

        # 2. Recognize Face
        (x, y, w, h) = faces[0]
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        
        prediction = knn.predict(resized_img)
        detected_name = prediction[0]

        # 3. Save to CSV
        date = datetime.now().strftime("%d-%m-%Y")
        timestamp = datetime.now().strftime("%H:%M:%S")
        file_path = f"Attendance/Attendance_{date}.csv"
        
        # Ensure directory exists
        if not os.path.exists("Attendance"):
            os.makedirs("Attendance")

        # Save record
        record = {'NAME': detected_name, 'TIME': timestamp, 'LOCATION': location_text}
        df = pd.DataFrame([record])
        
        if not os.path.isfile(file_path):
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, mode='a', header=False, index=False)

        return jsonify({
            "status": "success", 
            "name": detected_name,
            "date": date,
            "time": timestamp
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# --- Route 2: Registration (File Upload) ---
@app.route('/api/register', methods=['POST'])
def register():
    try:
        name = request.form.get('name')
        file = request.files.get('file')

        if not name or not file:
            return jsonify({"status": "error", "message": "Missing name or file"})

        # Read image file
        npimg = np.fromfile(file, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Detect and Process
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        facedetect = cv2.CascadeClassifier(f'{DATA_DIR}/haarcascade_frontalface_default.xml')
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return jsonify({"status": "error", "message": "No face found in upload"})

        (x, y, w, h) = faces[0]
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        # Save Data (Append 10 times for better training weight)
        names_list = []
        faces_list = None

        if os.path.exists(f'{DATA_DIR}/names.pkl'):
            with open(f'{DATA_DIR}/names.pkl', 'rb') as f:
                names_list = pickle.load(f)
        if os.path.exists(f'{DATA_DIR}/faces_data.pkl'):
            with open(f'{DATA_DIR}/faces_data.pkl', 'rb') as f:
                faces_list = pickle.load(f)

        # Adding sample 5 times to give it weight
        for _ in range(5):
            names_list.append(name)
            if faces_list is None:
                faces_list = resized_img
            else:
                faces_list = np.append(faces_list, resized_img, axis=0)

        with open(f'{DATA_DIR}/names.pkl', 'wb') as f:
            pickle.dump(names_list, f)
        with open(f'{DATA_DIR}/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_list, f)

        train_model() # Retrain immediately
        return jsonify({"status": "success", "message": f"User {name} Registered!"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    # '0.0.0.0' allows access from phone on same WiFi
    app.run(host='0.0.0.0', port=5000, debug=True)