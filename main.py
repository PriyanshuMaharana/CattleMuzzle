import os
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
import logging
import tempfile
from werkzeug.utils import secure_filename

MODEL_URL = "https://firebasestorage.googleapis.com/v0/b/test-75d65.appspot.com/o/cow_muzzle_feature_extractor.h5?alt=media&token=58fe8537-7fe1-45ac-a4e6-b92a3657c7ff"
MODEL_PATH = "cow_muzzle_feature_extractor.h5"

# Function to download the model if it doesn't exist locally
def download_model(url, save_path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Model downloaded and saved to {save_path}")
    except requests.RequestException as e:
        print(f"Error downloading model: {e}")
        raise

# Check if the model file exists, if not, download it
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    download_model(MODEL_URL, MODEL_PATH)

# Load the Keras model
model = load_model(MODEL_PATH)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
class Config:
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    INPUT_SHAPE = (71, 71, 3)

app.config.from_object(Config)

class H5ModelService:
    def __init__(self, model):
        self.model = model
        self.penultimate_layer_model = self.create_penultimate_layer_model()

    def create_penultimate_layer_model(self):
        """Create a submodel to extract features from the penultimate layer."""
        penultimate_layer = self.model.layers[-2].output  # Penultimate layer
        return tf.keras.Model(inputs=self.model.input, outputs=penultimate_layer)

    def preprocess_image(self, img_path):
        """Preprocess image for model input."""
        img = image.load_img(img_path, target_size=app.config['INPUT_SHAPE'][:2])
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return (img_array / 255.0).astype(np.float32)

    def extract_features(self, img_array):
        """Extract features using the penultimate layer model."""
        features = self.penultimate_layer_model.predict(img_array)
        return features.flatten()  # Flatten the features for output

# Initialize the H5 model service
h5_model_service = H5ModelService(model=model)

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200

@app.route('/extract_features', methods=['POST'])
def extract_features():
    """Extract features from uploaded image."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        # Create a temporary file for the image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            file.save(tmp_file.name)
            
            try:
                img_array = h5_model_service.preprocess_image(tmp_file.name)
                features = h5_model_service.extract_features(img_array)
                
                return jsonify({
                    'message': 'Features extracted successfully',
                    'features': features.tolist()
                })
            
            finally:
                # Clean up
                os.unlink(tmp_file.name)
                
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
