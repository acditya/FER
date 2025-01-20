# use python 3.11 or below pls 3.12 lost setuptools and you wil lost your mind 
# ~ adi from the past
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

model = load_model('cnn_emotion_detection.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_emotion(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return "No face detected"
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi_gray = roi_gray.astype('float32') / 255
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        
        prediction = model.predict(roi_gray)
        max_index = int(np.argmax(prediction))
        return emotion_labels[max_index]

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        emotion = predict_emotion(file_path)
        return jsonify({'emotion': emotion})

if __name__ == '__main__':
    # if not os.path.exists(app.config['UPLOAD_FOLDER']):
    #     os.makedirs(app.config['UPLOAD_FOLDER'])
    # app.run(debug=True)
    print(predict_emotion("Adi_DP.jpg"))