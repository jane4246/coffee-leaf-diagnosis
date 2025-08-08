from flask import Flask, request, jsonify
from flask_cors import CORS  # ⬅️ Add this
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # ⬅️ Enable CORS for all routes

MODEL_PATH = 'coffee_disease_model.h5'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model(MODEL_PATH)
classes = ['Healthy', 'Rust', 'Leaf Miner', 'Other']

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def home():
    return 'Coffee Leaf Diagnosis API is running.'

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        img = prepare_image(filepath)
        prediction = model.predict(img)[0]
        predicted_class = classes[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        return jsonify({'prediction': predicted_class, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
