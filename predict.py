from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load model and set up paths
MODEL_PATH = 'coffee_disease_model.h5'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = load_model(MODEL_PATH)

# Define class names
classes = ['Healthy', 'Rust', 'Leaf Miner', 'Other']  # Update as needed

# Image preprocessing function
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0,1]
    return np.expand_dims(img_array, axis=0)

# Prediction route
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
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
