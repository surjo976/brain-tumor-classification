from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)

# Load your trained model
# Replace 'path_to_your_model' with the actual path to your saved model
MODEL_PATH = 'F:/thesis/predict/optimized_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names
CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']

def preprocess_image(image):
    # Adjust these parameters according to your model's requirements
    target_size = (224, 224)  # Change this to match your model's input size
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to array and preprocess
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    
    # Normalize image (adjust according to your model's preprocessing)
    img_array = img_array / 255.0
    
    return img_array

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image file is present in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        
        # Check if the file is allowed
        if not allowed_file(image_file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Read and preprocess the image
        image = Image.open(io.BytesIO(image_file.read()))
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Prepare response
        response = {
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(CLASS_NAMES, predictions[0])
            }
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
