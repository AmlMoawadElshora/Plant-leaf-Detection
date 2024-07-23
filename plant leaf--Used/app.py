from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# Create Flask app
app = Flask(__name__)

# Load the model and labels
model = load_model('plant-leaf-model.h5')

labels = {0: 'Aml', 1: 'Alstonia Scholaris healthy (P2b)',
          2: 'Arjun diseased (P1a)', 3: 'Arjun healthy (P1b)', 4: 'Bael diseased (P4b)',
          5: 'Basil healthy (P8)', 6: 'Chinar diseased (P11b)', 7: 'Chinar healthy (P11a)',
          8: 'Gauva diseased (P3b)', 9: 'Gauva healthy (P3a)', 10: 'Jamun diseased (P5b)',
          11: 'Jamun healthy (P5a)', 12: 'Jatropha diseased (P6b)', 13: 'Jatropha healthy (P6a)',
          14: 'Lemon diseased (P10b)', 15: 'Lemon healthy (P10a)', 16: 'Mango diseased (P0b)',
          17: 'Mango healthy (P0a)', 18: 'Pomegranate diseased (P9b)', 19: 'Pomegranate healthy (P9a)',
          20: 'Pongamia Pinnata diseased (P7b)', 21: 'Pongamia Pinnata healthy (P7a)'}

# Define upload directory and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to render the upload form
@app.route('/')
def upload_form():
    return render_template('upload.html')

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Perform prediction
        img = image.load_img(filepath, target_size=(224, 224,3))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        prediction = model.predict(img_array)

        predicted_class_index = np.argmax(prediction)
        predicted_class = labels[predicted_class_index]
        
        return render_template('result.html', prediction=predicted_class)
        
    else:
        return "Invalid file format"

if __name__ == '__main__':
    app.run(debug=True)
