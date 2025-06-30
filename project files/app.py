import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
import io
import base64

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

RICE_CLASSES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

MODEL_PATH = 'rice.h5'
model = None

def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Model loaded successfully.")
        else:
            print(f"❌ Model file {MODEL_PATH} not found.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_rice_type(img):
    if model is None:
        print("❌ Model is None")
        return None, None

    try:
        processed_img = preprocess_image(img)
        print("✅ Processed shape:", processed_img.shape)

        predictions = model.predict(processed_img)
        print("✅ Predictions:", predictions)

        if predictions.shape[1] != len(RICE_CLASSES):
            print(f"❌ Prediction shape mismatch: {predictions.shape[1]} classes predicted but {len(RICE_CLASSES)} in label list")
            return None, None

        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        predicted_class = RICE_CLASSES[predicted_idx]

        print(f"✅ Final Prediction: {predicted_class} ({confidence:.2f})")

        return predicted_class, confidence

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        try:
            img = Image.open(io.BytesIO(file.read())).convert('RGB')

            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

            predicted_class, confidence = predict_rice_type(img)

            if predicted_class is None:
                flash('Prediction failed.')
                return redirect(url_for('index'))

            processed_img = preprocess_image(img)
            all_predictions = model.predict(processed_img)[0]
            results = {RICE_CLASSES[i]: float(all_predictions[i]) for i in range(len(RICE_CLASSES))}

            return render_template('result.html',
                                   predicted_class=predicted_class,
                                   confidence=confidence,
                                   results=results,
                                   image_data=img_base64)

        except Exception as e:
            flash(f'Error: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload PNG, JPG, JPEG, or GIF files.')
        return redirect(url_for('index'))

if __name__ == '__main__':
    load_model()
    app.run(debug=True)
